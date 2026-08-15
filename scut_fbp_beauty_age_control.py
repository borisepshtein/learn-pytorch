"""Does the CF pretty/average baseline hold up once age is controlled for?

Age was flagged twice as a plausible confound in the earlier experiments (a hairline/forehead
Grad-CAM hot spot on "average" predictions; visibly older faces among random "average" examples)
but never directly measured. This script estimates age for every CF image with an off-the-shelf
age-estimation model (DeepFace), then builds an age-matched subset -- for every age bin, keeps
equal numbers of "pretty" and "average" images, so the two label groups have (near-)identical age
distributions by construction -- and retrains the same ResNet18 recipe on it. A same-size random
(not age-matched) subsample is trained too, so any accuracy drop can be attributed to age-matching
specifically rather than just having fewer training images ("wrinkled apples to wrinkled apples").

Dataset: SCUT-FBP5500 (HCIILAB), non-commercial research only.
https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
"""

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

MIN_EPOCHS = 8
MAX_EPOCHS = 40
PATIENCE = 5
MIN_REL_IMPROVEMENT = 0.01
LEARNING_RATE = 1e-4

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.7, 0.15, 0.15
PRETTY_PERCENTILE = 50
AGE_BIN_WIDTH = 5  # years, for matching
SEED = 0

DATA_ROOT = './data'
GDRIVE_FILE_ID = '1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf'
ARCHIVE_PATH = os.path.join(DATA_ROOT, 'SCUT-FBP5500_v2.1.zip')
EXTRACT_DIR = os.path.join(DATA_ROOT, 'SCUT-FBP5500_v2')
IMAGES_DIR = os.path.join(EXTRACT_DIR, 'Images')
ALL_LABELS_PATH = os.path.join(EXTRACT_DIR, 'train_test_files', 'All_labels.txt')
AGE_CACHE_PATH = os.path.join(DATA_ROOT, 'scut_fbp5500_age_estimates.json')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


def ensure_dataset():
    os.makedirs(DATA_ROOT, exist_ok=True)
    if os.path.isdir(IMAGES_DIR):
        return
    if not os.path.isfile(ARCHIVE_PATH):
        try:
            import gdown
        except ImportError:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
            import gdown
        print(f'Downloading SCUT-FBP5500 dataset from Google Drive (id={GDRIVE_FILE_ID}, ~172MB) ...')
        try:
            gdown.download(id=GDRIVE_FILE_ID, output=ARCHIVE_PATH, quiet=False)
        except Exception as e:
            raise RuntimeError(
                f'Automatic download failed ({e}).\n'
                f'Please download SCUT-FBP5500_v2.1.zip manually (see '
                f'https://github.com/HCIILAB/SCUT-FBP5500-Database-Release for the Google Drive/Baidu links) '
                f'and place it at {ARCHIVE_PATH}.'
            ) from e
    print(f'Extracting {ARCHIVE_PATH} ...')
    with zipfile.ZipFile(ARCHIVE_PATH) as zf:
        zf.extractall(DATA_ROOT)
    if not os.path.isdir(IMAGES_DIR):
        raise RuntimeError(f'Extraction finished but {IMAGES_DIR} was not found; the archive layout may have changed.')


def build_items(prefix):
    records = []
    with open(ALL_LABELS_PATH) as f:
        for line in f:
            filename, score = line.split()
            if filename.startswith(prefix):
                records.append((filename, float(score)))
    scores = np.array([s for _, s in records])
    threshold = np.percentile(scores, PRETTY_PERCENTILE)
    return [(fn, score, int(score > threshold)) for fn, score in records], threshold


def ensure_deepface():
    try:
        from deepface import DeepFace
    except ImportError:
        print('Installing deepface (off-the-shelf age estimator) ...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'deepface'], check=True)
        from deepface import DeepFace
    return DeepFace


def estimate_ages(filenames, cache_key):
    """Returns {filename: estimated_age_or_None}, cached to disk since this is the slow step."""
    cache = {}
    if os.path.isfile(AGE_CACHE_PATH):
        with open(AGE_CACHE_PATH) as f:
            cache = json.load(f)
    if cache_key in cache and all(fn in cache[cache_key] for fn in filenames):
        print(f'Using cached age estimates for {cache_key} ({len(filenames)} images)')
        return cache[cache_key]

    DeepFace = ensure_deepface()
    ages = {}
    t0 = time.time()
    for i, fn in enumerate(filenames):
        path = os.path.join(IMAGES_DIR, fn)
        try:
            result = DeepFace.analyze(img_path=path, actions=['age'], enforce_detection=False, silent=True)
            ages[fn] = float(result[0]['age'] if isinstance(result, list) else result['age'])
        except Exception as e:
            print(f'  age estimation failed for {fn}: {e}')
            ages[fn] = None
        if (i + 1) % 100 == 0:
            print(f'  {cache_key}: age-estimated {i + 1}/{len(filenames)}  ({time.time() - t0:.0f}s)')

    cache[cache_key] = ages
    with open(AGE_CACHE_PATH, 'w') as f:
        json.dump(cache, f)
    return ages


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
        filename, score, label = self.items[idx]
        img = Image.open(os.path.join(IMAGES_DIR, filename)).convert('RGB')
        return self.transform(img), float(label)


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
    return {
        'n': len(labels),
        'accuracy': float(((probs > 0.5) == labels).mean()),
        'roc_auc': float(auc(fpr, tpr)),
    }, fpr, tpr


def run_condition(name, items):
    labels = np.array([label for _, _, label in items])
    train_items, temp_items = train_test_split(items, train_size=TRAIN_FRAC, stratify=labels, random_state=SEED)
    temp_labels = np.array([label for _, _, label in temp_items])
    val_items, test_items = train_test_split(
        temp_items, train_size=VAL_FRAC / (VAL_FRAC + TEST_FRAC), stratify=temp_labels, random_state=SEED)
    print(f'\n=== {name}: n={len(items)}  train={len(train_items)}  val={len(val_items)}  test={len(test_items)} ===')
    t0 = time.time()
    model, epochs_run, best_val_loss = train_model(train_items, val_items)
    metrics, fpr, tpr = evaluate(model, test_items)
    print(f'{name}: {epochs_run} epochs ({time.time() - t0:.0f}s)  accuracy={metrics["accuracy"]:.3f}  '
          f'roc_auc={metrics["roc_auc"]:.3f}')
    return {'name': name, 'n': len(items), 'n_train': len(train_items), 'epochs_run': epochs_run,
            'best_val_loss': float(best_val_loss), **metrics, 'fpr': fpr.tolist(), 'tpr': tpr.tolist()}


ensure_dataset()

cf_items, cf_threshold = build_items('CF')
af_items, af_threshold = build_items('AF')
print(f'CF: n={len(cf_items)}  score_threshold={cf_threshold:.3f}')

print('\nEstimating ages (CF) ...')
cf_ages_raw = estimate_ages([fn for fn, _, _ in cf_items], 'CF')
print('Estimating ages (AF, for cross-population age comparison) ...')
af_ages_raw = estimate_ages([fn for fn, _, _ in af_items], 'AF')

cf_data = [(fn, score, label, cf_ages_raw[fn]) for fn, score, label in cf_items if cf_ages_raw.get(fn) is not None]
af_ages = np.array([a for a in af_ages_raw.values() if a is not None])
n_failed = len(cf_items) - len(cf_data)
if n_failed:
    print(f'Age estimation failed for {n_failed}/{len(cf_items)} CF images; excluded from here on.')

ages = np.array([a for _, _, _, a in cf_data])
labels = np.array([label for _, _, label, _ in cf_data])
pretty_ages, average_ages = ages[labels == 1], ages[labels == 0]

print(f'\nCF age stats: mean={ages.mean():.1f}  std={ages.std():.1f}  '
      f'pretty_mean={pretty_ages.mean():.1f}  average_mean={average_ages.mean():.1f}')
print(f'Correlation(age, pretty-label) = {np.corrcoef(ages, labels)[0, 1]:.3f}')
print(f'AF age stats (for comparison): mean={af_ages.mean():.1f}  std={af_ages.std():.1f}')

# --- Build an age-matched subset: equal pretty/average counts within each age bin ---
bin_edges = np.arange(np.floor(ages.min() / AGE_BIN_WIDTH) * AGE_BIN_WIDTH,
                       ages.max() + AGE_BIN_WIDTH, AGE_BIN_WIDTH)
bin_idx = np.digitize(ages, bin_edges)

rng = np.random.default_rng(SEED)
matched_positions = []
for b in np.unique(bin_idx):
    in_bin = np.where(bin_idx == b)[0]
    pos = in_bin[labels[in_bin] == 1]
    neg = in_bin[labels[in_bin] == 0]
    n = min(len(pos), len(neg))
    if n == 0:
        continue
    matched_positions.extend(rng.choice(pos, n, replace=False))
    matched_positions.extend(rng.choice(neg, n, replace=False))
matched_positions = np.array(sorted(matched_positions))
matched_items = [(cf_data[i][0], cf_data[i][1], cf_data[i][2]) for i in matched_positions]
n_matched = len(matched_items)
print(f'\nAge-matched subset: n={n_matched}  (from {len(cf_data)}, bin width={AGE_BIN_WIDTH}y)')

# --- Same-size random (label-stratified, NOT age-matched) control ---
all_positions = np.arange(len(cf_data))
random_positions, _ = train_test_split(all_positions, train_size=n_matched, stratify=labels, random_state=SEED)
random_items = [(cf_data[i][0], cf_data[i][1], cf_data[i][2]) for i in random_positions]

results = [
    run_condition('Full CF (age-unmatched baseline)', [(fn, s, l) for fn, s, l, _ in cf_data]),
    run_condition(f'Random subsample (n={n_matched}, not age-matched)', random_items),
    run_condition(f'Age-matched subsample (n={n_matched})', matched_items),
]

# --- Visualization: age distributions before/after matching, ROC comparison, accuracy/AUC bars ---
fig = plt.figure(figsize=(11, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.2, 1])

ax_hist = fig.add_subplot(gs[0])
bins = np.arange(ages.min(), ages.max() + 2, 2)
ax_hist.hist(pretty_ages, bins=bins, alpha=0.6, label=f'Pretty (mean={pretty_ages.mean():.1f})', color='tab:orange')
ax_hist.hist(average_ages, bins=bins, alpha=0.6, label=f'Average (mean={average_ages.mean():.1f})', color='tab:blue')
ax_hist.set_xlabel('Estimated age (years)')
ax_hist.set_ylabel('Count')
ax_hist.set_title('CF age distribution by label, before matching')
ax_hist.legend()

matched_ages = ages[matched_positions]
matched_labels = labels[matched_positions]
ax_hist2 = fig.add_subplot(gs[1])
ax_hist2.hist(matched_ages[matched_labels == 1], bins=bins, alpha=0.6,
              label=f'Pretty (mean={matched_ages[matched_labels == 1].mean():.1f})', color='tab:orange')
ax_hist2.hist(matched_ages[matched_labels == 0], bins=bins, alpha=0.6,
              label=f'Average (mean={matched_ages[matched_labels == 0].mean():.1f})', color='tab:blue')
ax_hist2.set_xlabel('Estimated age (years)')
ax_hist2.set_ylabel('Count')
ax_hist2.set_title(f'Age distribution after matching (n={n_matched})')
ax_hist2.legend()

ax_bar = fig.add_subplot(gs[2])
x = np.arange(len(results))
width = 0.35
accs = [r['accuracy'] for r in results]
aucs = [r['roc_auc'] for r in results]
ax_bar.bar(x - width / 2, accs, width, label='Accuracy', color='tab:blue')
ax_bar.bar(x + width / 2, aucs, width, label='ROC AUC', color='tab:orange')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([f"{r['name']}\n(n={r['n']})" for r in results], fontsize=8)
ax_bar.set_ylim(0, 1.05)
ax_bar.set_title('Accuracy / ROC AUC: full vs. random-subsample vs. age-matched')
ax_bar.legend(loc='lower right')

fig.tight_layout()
os.makedirs('results', exist_ok=True)
fig.savefig('results/scut_fbp_beauty_age_control.png', dpi=150)
print('\nSaved visualization to results/scut_fbp_beauty_age_control.png')

metrics = {
    'age_bin_width': AGE_BIN_WIDTH,
    'n_cf_total': len(cf_items),
    'n_cf_age_estimated': len(cf_data),
    'cf_age_mean': float(ages.mean()),
    'cf_age_std': float(ages.std()),
    'cf_pretty_age_mean': float(pretty_ages.mean()),
    'cf_average_age_mean': float(average_ages.mean()),
    'age_label_correlation': float(np.corrcoef(ages, labels)[0, 1]),
    'af_age_mean': float(af_ages.mean()),
    'af_age_std': float(af_ages.std()),
    'n_matched': n_matched,
    'conditions': [{k: v for k, v in r.items() if k not in ('fpr', 'tpr')} for r in results],
}
with open('results/scut_fbp_beauty_age_control_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
log_experiment('scut_fbp_beauty_age_control', metrics)
