"""Combines three independently-rated face-attractiveness datasets -- SCUT-FBP5500, the Chicago
Face Database (CFD), and the Face Research Lab London Set -- each with a DIFFERENT rater pool and
photographic style, unlike the SCUT-only cross-race experiments earlier where every rater was
from the same 60-person Asian rater pool. Restricted to female subjects for direct comparability
with the rest of this project. Every training image is a face-only crop (SCUT: its own 86-point
landmarks; CFD/London: YuNet deep-learning face detection via cv2, cached) to avoid the hair-region shortcut
confirmed in scut_fbp_beauty_cross_race_transfer_no_hair.py.

Validation: leave-one-source-out cross-validation (train on 2 sources, evaluate in-domain on a
held-out slice of those 2 plus out-of-domain on the fully-held-out 3rd source) for all 3
combinations, then one final model trained on all three sources combined -- the actual "best
available" classifier this project produces, per the user's explicit ask for a classifier that
doesn't just pick up wrong cues.

Data access:
  - SCUT-FBP5500: downloaded automatically (Google Drive, public link).
  - Face Research Lab London Set: downloaded automatically (figshare, CC-BY, no gate).
  - CFD: gated behind a personal access request (https://www.chicagofaces.org/download/) and not
    redistributable. Upload the resulting zip to your own Google Drive at
    'MyDrive/beauty_classifier_data/CFD.zip' before running this in Colab; this script mounts
    your Drive to read it rather than downloading it from any URL.
"""

import csv
import json
import os
import subprocess
import sys
import time
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from experiment_log import log_experiment

IMG_SIZE = 224
BATCH_SIZE = 32
N_EXAMPLES = 4

MIN_EPOCHS = 8
MAX_EPOCHS = 40
PATIENCE = 5
MIN_REL_IMPROVEMENT = 0.01
LEARNING_RATE = 1e-4

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.7, 0.15, 0.15
PRETTY_PERCENTILE = 50
SEED = 0

DATA_ROOT = './data'

# --- SCUT-FBP5500 ---
SCUT_GDRIVE_FILE_ID = '1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf'
SCUT_ARCHIVE_PATH = os.path.join(DATA_ROOT, 'SCUT-FBP5500_v2.1.zip')
SCUT_EXTRACT_DIR = os.path.join(DATA_ROOT, 'SCUT-FBP5500_v2')
SCUT_IMAGES_DIR = os.path.join(SCUT_EXTRACT_DIR, 'Images')
SCUT_LANDMARKS_DIR = os.path.join(SCUT_EXTRACT_DIR, 'facial landmark')
SCUT_ALL_LABELS_PATH = os.path.join(SCUT_EXTRACT_DIR, 'train_test_files', 'All_labels.txt')
LANDMARK_MARGIN = 0.05

# --- CFD (gated, via Google Drive) ---
CFD_DRIVE_ZIP = '/content/drive/MyDrive/beauty_classifier_data/CFD.zip'
CFD_EXTRACT_DIR = os.path.join(DATA_ROOT, 'CFD')

# --- Face Research Lab London Set ---
LONDON_EXTRACT_DIR = os.path.join(DATA_ROOT, 'London')
LONDON_INFO_URL = 'https://ndownloader.figshare.com/files/27397184'
LONDON_RATINGS_URL = 'https://ndownloader.figshare.com/files/8542045'
LONDON_IMAGES_URL = 'https://ndownloader.figshare.com/files/8541961'

FACE_BBOX_MARGIN = 0.5  # YuNet's raw box is tight (eyes-to-chin); 0.5 gives eyebrow/hairline-to-chin
FACE_DETECT_SCORE_THRESHOLD = 0.3  # verified real faces at 0.6 threshold scored as low as 0.54
YUNET_MODEL_URL = ('https://github.com/opencv/opencv_zoo/raw/main/models/'
                    'face_detection_yunet/face_detection_yunet_2023mar.onnx')
YUNET_MODEL_PATH = os.path.join(DATA_ROOT, 'face_detection_yunet_2023mar.onnx')
BBOX_CACHE_PATH = os.path.join(DATA_ROOT, 'yunet_face_bboxes.json')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


def pip_install(pkg):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)


# ---------------- SCUT-FBP5500 ----------------
def ensure_scut():
    os.makedirs(DATA_ROOT, exist_ok=True)
    if os.path.isdir(SCUT_IMAGES_DIR):
        return
    if not os.path.isfile(SCUT_ARCHIVE_PATH):
        try:
            import gdown
        except ImportError:
            pip_install('gdown')
            import gdown
        print(f'Downloading SCUT-FBP5500 (id={SCUT_GDRIVE_FILE_ID}, ~172MB) ...')
        gdown.download(id=SCUT_GDRIVE_FILE_ID, output=SCUT_ARCHIVE_PATH, quiet=False)
    print(f'Extracting {SCUT_ARCHIVE_PATH} ...')
    with zipfile.ZipFile(SCUT_ARCHIVE_PATH) as zf:
        zf.extractall(DATA_ROOT)


def scut_landmark_bbox(filename):
    import struct
    pts_path = os.path.join(SCUT_LANDMARKS_DIR, os.path.splitext(filename)[0] + '.pts')
    try:
        with open(pts_path, 'rb') as f:
            data = f.read()
        points = struct.unpack('i172f', data)
        xs, ys = points[1::2], points[2::2]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        mw, mh = (right - left) * LANDMARK_MARGIN, (bottom - top) * LANDMARK_MARGIN
        return left - mw, top - mh, right + mw, bottom + mh
    except Exception:
        return None


def build_scut_items():
    """Pools CF+AF (Caucasian + Asian female) into a single 'SCUT' source with one shared
    median-split threshold, since here SCUT represents one rater culture, not two subsets to
    compare against each other."""
    records = []
    with open(SCUT_ALL_LABELS_PATH) as f:
        for line in f:
            filename, score = line.split()
            if filename.startswith('CF') or filename.startswith('AF'):
                records.append((filename, float(score)))
    scores = np.array([s for _, s in records])
    threshold = np.percentile(scores, PRETTY_PERCENTILE)
    items = []
    for fn, score in records:
        path = os.path.join(SCUT_IMAGES_DIR, fn)
        bbox = scut_landmark_bbox(fn)
        items.append({'path': path, 'score': score, 'label': int(score > threshold),
                       'source': 'SCUT', 'bbox': bbox})
    print(f'SCUT: n={len(items)}  threshold={threshold:.3f}')
    return items


# ---------------- CFD ----------------
def ensure_cfd():
    """Note: does NOT call drive.mount() itself -- that only works from an actual notebook cell
    (it needs the interactive Colab kernel connection), not from a script run via `!python ...`,
    which executes as a plain subprocess with no kernel to talk to. Drive must already be mounted
    (see the notebook cell before this script's cell) before this function is called."""
    os.makedirs(DATA_ROOT, exist_ok=True)
    if os.path.isdir(CFD_EXTRACT_DIR):
        return True
    if not os.path.isdir('/content/drive/MyDrive'):
        print('Google Drive is not mounted. Run the "mount Google Drive" cell above (must be its own '
              'notebook cell, not this script) and re-run. Skipping CFD for this run.')
        return False
    if not os.path.isfile(CFD_DRIVE_ZIP):
        print(f'CFD not found at {CFD_DRIVE_ZIP} in your Google Drive -- skipping CFD for this run.\n'
              f'Request access at https://www.chicagofaces.org/download/ and upload the zip there to include it.')
        return False
    print(f'Extracting {CFD_DRIVE_ZIP} ...')
    os.makedirs(CFD_EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(CFD_DRIVE_ZIP) as zf:
        zf.extractall(CFD_EXTRACT_DIR)
    return True


def _normalize(s):
    return ''.join(c for c in s.upper() if c.isalnum())


def build_cfd_items():
    """Female individuals across CFD core + CFD-MR (multiracial) + CFD-INDIA, each rated on a
    1-7 scale by their own norming study; one shared median-split threshold across all of CFD."""
    try:
        import openpyxl
    except ImportError:
        pip_install('openpyxl')
        import openpyxl

    root = None
    for dirpath, dirnames, filenames in os.walk(CFD_EXTRACT_DIR):
        if any(f.endswith('Codebook.xlsx') for f in filenames):
            root = dirpath
            break
    if root is None:
        raise RuntimeError(f'Could not find the CFD norming spreadsheet under {CFD_EXTRACT_DIR}.')
    xlsx_path = next(os.path.join(root, f) for f in os.listdir(root) if f.endswith('Codebook.xlsx'))
    img_root = os.path.join(root, 'Images')

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sheets = ['CFD U.S. Norming Data', 'CFD-MR U.S. Norming Data', 'CFD-I INDIA Norming Data']
    subdirs = ['CFD', 'CFD-MR', 'CFD-INDIA']

    all_records = []  # (model, gender, ethnicity, attractiveness, subdir)
    for sheet, subdir in zip(sheets, subdirs):
        ws = wb[sheet]
        rows = list(ws.iter_rows(min_row=8, values_only=True))
        names = rows[0]
        model_i, eth_i, gender_i, attr_i = (names.index(c) for c in
                                             ('Model', 'EthnicitySelf', 'GenderSelf', 'Attractive'))
        for r in rows[1:]:
            if r[model_i] is None or r[attr_i] is None:
                continue
            if r[gender_i] != 'F':
                continue
            all_records.append((str(r[model_i]), r[eth_i], r[attr_i], subdir))

    # index every neutral ("-N.jpg") image file per subdir, then match by normalized-ID prefix
    file_index = {}
    for subdir in subdirs:
        image_dir = os.path.join(img_root, subdir)
        idx = {}
        for dirpath, _, files in os.walk(image_dir):
            for fn in files:
                if fn.upper().endswith('-N.JPG'):
                    idx[_normalize(fn[:-4])] = os.path.join(dirpath, fn)
        file_index[subdir] = idx

    scores = np.array([a for _, _, a, _ in all_records])
    threshold = np.percentile(scores, PRETTY_PERCENTILE)

    items, unmatched = [], 0
    for model, eth, attr, subdir in all_records:
        norm_model = _normalize(model)
        hit = next((path for key, path in file_index[subdir].items()
                    if key.startswith('CFD' + norm_model) or key.startswith(norm_model)), None)
        if hit is None:
            unmatched += 1
            continue
        items.append({'path': hit, 'score': float(attr), 'label': int(attr > threshold),
                       'source': 'CFD', 'ethnicity': eth, 'bbox': None})
    print(f'CFD: n={len(items)}  threshold={threshold:.3f}  unmatched={unmatched}')
    return items


# ---------------- Face Research Lab London Set ----------------
def ensure_london():
    os.makedirs(LONDON_EXTRACT_DIR, exist_ok=True)
    info_path = os.path.join(LONDON_EXTRACT_DIR, 'info.csv')
    ratings_path = os.path.join(LONDON_EXTRACT_DIR, 'ratings.csv')
    images_zip = os.path.join(LONDON_EXTRACT_DIR, 'neutral_front.zip')
    images_dir = os.path.join(LONDON_EXTRACT_DIR, 'neutral_front')
    if not os.path.isfile(info_path):
        _urlretrieve(LONDON_INFO_URL, info_path)
    if not os.path.isfile(ratings_path):
        _urlretrieve(LONDON_RATINGS_URL, ratings_path)
    if not os.path.isdir(images_dir):
        _urlretrieve(LONDON_IMAGES_URL, images_zip)
        with zipfile.ZipFile(images_zip) as zf:
            zf.extractall(images_dir)
    return info_path, ratings_path, images_dir


def _urlretrieve(url, path):
    import urllib.request
    print(f'Downloading {os.path.basename(path)} ...')
    urllib.request.urlretrieve(url, path)


def build_london_items():
    """All female individuals; attractiveness = mean across raters (wide-format ratings CSV, one
    column per face_id). One shared median-split threshold across London."""
    info_path, ratings_path, images_dir = ensure_london()

    genders = {}
    with open(info_path) as f:
        for row in csv.DictReader(f):
            genders[row['face_id']] = row['face_gender']

    with open(ratings_path) as f:
        reader = csv.DictReader(f)
        face_cols = [c for c in reader.fieldnames if c.startswith('X')]
        sums = {c: 0.0 for c in face_cols}
        counts = {c: 0 for c in face_cols}
        for row in reader:
            for c in face_cols:
                v = row.get(c, '').strip()
                if v:
                    sums[c] += float(v)
                    counts[c] += 1
    mean_scores = {c[1:]: sums[c] / counts[c] for c in face_cols if counts[c] > 0}

    female_scores = {fid: s for fid, s in mean_scores.items() if genders.get(fid) == 'female'}
    threshold = np.percentile(list(female_scores.values()), PRETTY_PERCENTILE)

    img_root = None
    for dirpath, _, files in os.walk(images_dir):
        # zip extraction leaves a __MACOSX junk folder of AppleDouble resource-fork files
        # (e.g. "._001_03.jpg") alongside the real one; skip it explicitly.
        if '__MACOSX' in dirpath:
            continue
        if any(f.endswith('.jpg') and not f.startswith('._') for f in files):
            img_root = dirpath
            break

    items = []
    for fid, score in female_scores.items():
        # some entries have a stray non-image companion file (e.g. "001_03.tem") alongside
        # the real "001_03.jpg"; restrict to actual image extensions.
        candidates = [f for f in os.listdir(img_root)
                      if f.startswith(fid + '_') and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not candidates:
            continue
        path = os.path.join(img_root, candidates[0])
        items.append({'path': path, 'score': score, 'label': int(score > threshold),
                       'source': 'London', 'bbox': None})
    print(f'London: n={len(items)}  threshold={threshold:.3f}')
    return items


# ---------------- Face-crop (CFD + London) ----------------
def ensure_yunet_model():
    if not os.path.isfile(YUNET_MODEL_PATH):
        import urllib.request
        print('Downloading YuNet face detection model (~230KB) ...')
        urllib.request.urlretrieve(YUNET_MODEL_URL, YUNET_MODEL_PATH)


def compute_bboxes(items):
    """Fills in item['bbox'] for CFD/London items via YuNet (cv2.FaceDetectorYN), a lightweight
    deep-learning face detector bundled with OpenCV's objdetect module, cached to disk since
    re-running detection every epoch would be wasteful and this is the slow step.

    Two other face detectors were tried first and both broke on this Colab image: DeepFace's
    'opencv' backend crashed with a missing cv2.CascadeClassifier attribute, then plain OpenCV
    Haar cascades (chosen as the fallback) worked but are meaningfully less accurate than a real
    detector; MediaPipe (mp.solutions.face_detection) crashed with a missing 'solutions' attribute,
    a known unresolved upstream bug. YuNet gives actual DL-quality detection while still requiring
    no extra pip install (only cv2, already needed) -- just one small (~230KB) .onnx model file
    fetched at runtime, same pattern as every other dataset asset in this script.

    Threshold note: at the OpenCV-recommended default score_threshold=0.6, 70/410 real CFD faces
    were missed (verified locally, roughly proportional across all ethnicity groups: not a
    detector bias, since every genuine face scored 0.54-0.59 and 0.3 recovers all 410/410 with no
    spurious detections). FACE_BBOX_MARGIN is larger than the earlier MediaPipe/Haar attempts
    because YuNet's raw box is tighter (eyes-to-chin rather than eyebrow-to-chin).
    """
    ensure_yunet_model()
    cache = {}
    if os.path.isfile(BBOX_CACHE_PATH):
        with open(BBOX_CACHE_PATH) as f:
            cache = json.load(f)

    to_compute = [it for it in items if it['bbox'] is None and it['path'] not in cache]
    if to_compute:
        import cv2
        detector = cv2.FaceDetectorYN_create(YUNET_MODEL_PATH, '', (320, 320),
                                              score_threshold=FACE_DETECT_SCORE_THRESHOLD)
        t0 = time.time()
        for i, it in enumerate(to_compute):
            img = Image.open(it['path']).convert('RGB')
            arr = np.array(img)[:, :, ::-1]  # RGB -> BGR for opencv
            h, w = arr.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(arr)
            if faces is not None and len(faces):
                x, y, fw, fh = max(faces, key=lambda f: f[14])[:4]  # highest-confidence face
                mx, my = fw * FACE_BBOX_MARGIN, fh * FACE_BBOX_MARGIN
                cache[it['path']] = [max(0, float(x - mx)), max(0, float(y - my)),
                                      min(w, float(x + fw + mx)), min(h, float(y + fh + my))]
            else:
                cache[it['path']] = None
            if (i + 1) % 200 == 0:
                print(f'  face-detected {i + 1}/{len(to_compute)}  ({time.time() - t0:.0f}s)')
        with open(BBOX_CACHE_PATH, 'w') as f:
            json.dump(cache, f)

    for it in items:
        if it['bbox'] is None:
            it['bbox'] = cache.get(it['path'])
    n_missing = sum(1 for it in items if it['bbox'] is None)
    if n_missing:
        print(f'  {n_missing}/{len(items)} images had no detected face; using full image for those.')
    return items


def face_crop(img, bbox):
    if bbox is None:
        return img
    w, h = img.size
    left, top, right, bottom = bbox
    left, top = max(0, left), max(0, top)
    right, bottom = min(w, right), min(h, bottom)
    if right <= left or bottom <= top:
        return img
    return img.crop((left, top, right, bottom))


# ---------------- Model / training (shared) ----------------
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class FaceDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it['path']).convert('RGB')
        img = face_crop(img, it['bbox'])
        return self.transform(img), float(it['label'])


def train_model(train_items, val_items):
    train_loader = DataLoader(FaceDataset(train_items, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FaceDataset(val_items, eval_transform), batch_size=BATCH_SIZE, shuffle=False)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0
    epoch = 0

    while epoch < MAX_EPOCHS:
        epoch += 1
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images).squeeze(1), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                val_running_loss += criterion(model(images).squeeze(1), targets).item()
        val_loss = val_running_loss / len(val_loader)

        if val_loss < best_val_loss * (1 - MIN_REL_IMPROVEMENT):
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch >= MIN_EPOCHS and epochs_without_improvement >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, epoch, best_val_loss


def evaluate(model, items):
    """Reports accuracy at two thresholds: the naive fixed 0.5 cutoff, and each evaluation set's
    own median predicted score -- mirroring how labels themselves are defined via each source's
    own score median (build_scut_items/build_cfd_items/build_london_items), rather than a single
    global cutoff. Earlier cross-domain results repeatedly showed AUC (rank-ordering) transferring
    far better than fixed-threshold accuracy: the model's raw score distribution shifts between
    domains even when its ranking of faces within a domain stays informative, so a threshold tuned
    on the source domain can land in the wrong place on the target domain's score distribution.
    Median-threshold accuracy tests directly whether that's the whole story."""
    loader = DataLoader(FaceDataset(items, eval_transform), batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            probs.append(torch.sigmoid(model(images).squeeze(1)).cpu())
            labels.append(targets)
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy().astype(int)
    fpr, tpr, _ = roc_curve(labels, probs)
    own_median = float(np.median(probs))
    metrics = {
        'n': len(labels),
        'accuracy': float(((probs > 0.5) == labels).mean()),
        'accuracy_own_median_threshold': float(((probs > own_median) == labels).mean()),
        'own_median_threshold': own_median,
        'roc_auc': float(auc(fpr, tpr)),
    }
    return metrics, fpr, tpr


def gradcam_heatmap(model, target_layer, input_tensor):
    activations, gradients = {}, {}

    def forward_hook(module, inp, out):
        activations['value'] = out

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    h_fwd = target_layer.register_forward_hook(forward_hook)
    h_bwd = target_layer.register_full_backward_hook(backward_hook)
    model.zero_grad()
    output = model(input_tensor.unsqueeze(0).to(device)).squeeze()
    output.backward()
    h_fwd.remove()
    h_bwd.remove()

    acts = activations['value'][0]
    grads = gradients['value'][0]
    weights = grads.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
    cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(cam_img) / 255.0


# ---------------- Assemble sources ----------------
ensure_scut()
sources = {'SCUT': build_scut_items()}

have_cfd = ensure_cfd()
if have_cfd:
    sources['CFD'] = build_cfd_items()
else:
    print('Proceeding without CFD for this run.')

sources['London'] = build_london_items()

for name in ('CFD', 'London'):
    if name in sources:
        print(f'Computing/loading face bounding boxes for {name} ...')
        sources[name] = compute_bboxes(sources[name])

active_sources = list(sources.keys())
print(f'\nActive sources: {active_sources}')
for name, items in sources.items():
    labels = [it['label'] for it in items]
    print(f'  {name}: n={len(items)}  pretty={sum(labels)}  average={len(labels) - sum(labels)}')

# ---------------- Leave-one-source-out CV (only if we have all 3) ----------------
loso_results = []
if len(active_sources) >= 3:
    for held_out in active_sources:
        train_sources = [s for s in active_sources if s != held_out]
        pool = [it for s in train_sources for it in sources[s]]
        pool_labels = np.array([it['label'] for it in pool])
        train_items, temp_items = train_test_split(pool, train_size=TRAIN_FRAC, stratify=pool_labels,
                                                     random_state=SEED)
        temp_labels = np.array([it['label'] for it in temp_items])
        val_items, test_items = train_test_split(
            temp_items, train_size=VAL_FRAC / (VAL_FRAC + TEST_FRAC), stratify=temp_labels, random_state=SEED)

        print(f'\n=== Leave-{held_out}-out: train on {train_sources} (n={len(pool)}), '
              f'test on {held_out} (n={len(sources[held_out])}) ===')
        t0 = time.time()
        model, epochs_run, best_val_loss = train_model(train_items, val_items)
        in_metrics, in_fpr, in_tpr = evaluate(model, test_items)
        out_metrics, out_fpr, out_tpr = evaluate(model, sources[held_out])
        print(f'  {epochs_run} epochs ({time.time() - t0:.0f}s)  '
              f'in-domain acc={in_metrics["accuracy"]:.3f} (median-thresh={in_metrics["accuracy_own_median_threshold"]:.3f}) '
              f'auc={in_metrics["roc_auc"]:.3f}  '
              f'out-of-domain({held_out}) acc={out_metrics["accuracy"]:.3f} '
              f'(median-thresh={out_metrics["accuracy_own_median_threshold"]:.3f}) auc={out_metrics["roc_auc"]:.3f}')
        loso_results.append({
            'held_out': held_out, 'train_sources': train_sources, 'n_train': len(train_items),
            'epochs_run': epochs_run, 'best_val_loss': float(best_val_loss),
            'in_domain': in_metrics, 'out_of_domain': out_metrics,
            'in_domain_fpr': in_fpr.tolist(), 'in_domain_tpr': in_tpr.tolist(),
            'out_domain_fpr': out_fpr.tolist(), 'out_domain_tpr': out_tpr.tolist(),
        })
else:
    print('\nFewer than 3 sources available -- skipping leave-one-source-out CV this run.')

# ---------------- Final model: all available sources combined ----------------
all_items = [it for s in active_sources for it in sources[s]]
all_labels = np.array([it['label'] for it in all_items])
train_items, temp_items = train_test_split(all_items, train_size=TRAIN_FRAC, stratify=all_labels, random_state=SEED)
temp_labels = np.array([it['label'] for it in temp_items])
val_items, test_items = train_test_split(
    temp_items, train_size=VAL_FRAC / (VAL_FRAC + TEST_FRAC), stratify=temp_labels, random_state=SEED)

print(f'\n=== Final combined model: sources={active_sources}  n={len(all_items)} '
      f'train={len(train_items)} val={len(val_items)} test={len(test_items)} ===')
t0 = time.time()
final_model, final_epochs, final_best_val_loss = train_model(train_items, val_items)
final_metrics, final_fpr, final_tpr = evaluate(final_model, test_items)
print(f'Final model: {final_epochs} epochs ({time.time() - t0:.0f}s)  '
      f'accuracy={final_metrics["accuracy"]:.3f} '
      f'(median-thresh={final_metrics["accuracy_own_median_threshold"]:.3f})  '
      f'roc_auc={final_metrics["roc_auc"]:.3f}')

per_source_final = {}
for name in active_sources:
    src_test = [it for it in test_items if it['source'] == name]
    if len(src_test) >= 10:
        m, _, _ = evaluate(final_model, src_test)
        per_source_final[name] = m
        print(f'  final model on {name}-only test slice (n={m["n"]}): '
              f'accuracy={m["accuracy"]:.3f} (median-thresh={m["accuracy_own_median_threshold"]:.3f})  '
              f'roc_auc={m["roc_auc"]:.3f}')

# ---------------- Visualization ----------------
n_rows = len(loso_results) + 3  # each LOSO ROC, final ROC, calibration bar chart, Grad-CAM strip
fig = plt.figure(figsize=(11, 4.2 * n_rows))
gs = fig.add_gridspec(n_rows, N_EXAMPLES * 2)

row = 0
for r in loso_results:
    ax = fig.add_subplot(gs[row, :])
    ax.plot(r['in_domain_fpr'], r['in_domain_tpr'], color='tab:blue',
            label=f"In-domain (AUC={r['in_domain']['roc_auc']:.3f})")
    ax.plot(r['out_domain_fpr'], r['out_domain_tpr'], color='tab:red',
            label=f"Out-of-domain: {r['held_out']} (AUC={r['out_of_domain']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=0.8)
    ax.set_title(f"Trained on {r['train_sources']}, held out {r['held_out']}")
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend(loc='lower right', fontsize=8)
    row += 1

ax_final = fig.add_subplot(gs[row, :])
ax_final.plot(final_fpr, final_tpr, color='tab:green', label=f'All sources combined (AUC={final_metrics["roc_auc"]:.3f})')
ax_final.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=0.8)
ax_final.set_title(f'Final model ({"+".join(active_sources)}) test ROC')
ax_final.set_xlabel('False positive rate')
ax_final.set_ylabel('True positive rate')
ax_final.legend(loc='lower right', fontsize=8)
row += 1

ax_cal = fig.add_subplot(gs[row, :])
cal_labels, cal_fixed, cal_median = [], [], []
for r in loso_results:
    cal_labels.append(f"out: {r['held_out']}")
    cal_fixed.append(r['out_of_domain']['accuracy'])
    cal_median.append(r['out_of_domain']['accuracy_own_median_threshold'])
for name, m in per_source_final.items():
    cal_labels.append(f"final: {name}")
    cal_fixed.append(m['accuracy'])
    cal_median.append(m['accuracy_own_median_threshold'])
x = np.arange(len(cal_labels))
width = 0.35
ax_cal.bar(x - width / 2, cal_fixed, width, label='Accuracy @ fixed 0.5', color='tab:red')
ax_cal.bar(x + width / 2, cal_median, width, label='Accuracy @ own median threshold', color='tab:green')
ax_cal.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance')
ax_cal.set_xticks(x)
ax_cal.set_xticklabels(cal_labels, fontsize=8)
ax_cal.set_ylim(0, 1.05)
ax_cal.set_title('Calibration test: does a per-domain median threshold recover the accuracy AUC implies?')
ax_cal.legend(loc='lower right', fontsize=8)
row += 1

loader = DataLoader(FaceDataset(test_items, eval_transform), batch_size=BATCH_SIZE, shuffle=False)
final_model.eval()
all_probs = []
with torch.no_grad():
    for images, _ in loader:
        all_probs.append(torch.sigmoid(final_model(images.to(device)).squeeze(1)).cpu())
all_probs = torch.cat(all_probs).numpy()
top_idx = np.argsort(-all_probs)[:N_EXAMPLES]
bottom_idx = np.argsort(all_probs)[:N_EXAMPLES]
gradcam_layer = final_model.layer4[-1]

for col, idx in enumerate(list(top_idx) + list(bottom_idx)):
    it = test_items[idx]
    img = Image.open(it['path']).convert('RGB')
    img = face_crop(img, it['bbox'])
    face = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    cam = gradcam_heatmap(final_model, gradcam_layer, eval_transform(img))
    ax = fig.add_subplot(gs[row, col])
    ax.imshow(face)
    ax.imshow(cam, cmap='jet', alpha=0.45)
    ax.set_xticks([])
    ax.set_yticks([])
    true_label = 'pretty' if it['label'] == 1 else 'average'
    ax.set_title(f'{all_probs[idx]:.2f} ({it["source"]}, {true_label})', fontsize=7)

fig.tight_layout()
os.makedirs('results', exist_ok=True)
fig.savefig('results/beauty_classifier_multisource.png', dpi=150)
print('\nSaved visualization to results/beauty_classifier_multisource.png')

metrics = {
    'active_sources': active_sources,
    'source_counts': {name: len(items) for name, items in sources.items()},
    'leave_one_source_out': [{k: v for k, v in r.items()
                               if k not in ('in_domain_fpr', 'in_domain_tpr', 'out_domain_fpr', 'out_domain_tpr')}
                              for r in loso_results],
    'final_model': {
        'n_total': len(all_items), 'n_train': len(train_items), 'epochs_run': final_epochs,
        'best_val_loss': float(final_best_val_loss), **final_metrics,
        'per_source_test': per_source_final,
    },
}
with open('results/beauty_classifier_multisource_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
log_experiment('beauty_classifier_multisource', metrics)
