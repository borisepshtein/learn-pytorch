"""Does a beauty classifier trained on one race/gender subset transfer to another?

Trains a ResNet18 on the Caucasian-female pretty/average distinction (median split, same
recipe as scut_fbp_beauty_classifier.py) and evaluates it both in-domain (held-out CF faces)
and cross-domain (the full Asian-female subset, labeled by ITS OWN median split) -- then
repeats in the opposite direction (train on AF, test on CF). If beauty judgments were purely
a within-population/cultural construct, cross-domain performance should collapse toward
chance; if the same 60 SCUT raters' judgments rest on visual cues that hold up across race,
cross-domain AUC should stay well above 0.5.

Dataset: SCUT-FBP5500 (HCIILAB), non-commercial research use only.
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
N_EXAMPLES = 2  # per extreme (highest/lowest predicted score) in the cross-domain Grad-CAM strip

MIN_EPOCHS = 8
MAX_EPOCHS = 40
PATIENCE = 5
MIN_REL_IMPROVEMENT = 0.01
LEARNING_RATE = 1e-4

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.7, 0.15, 0.15
PRETTY_PERCENTILE = 50

TRANSFER_PAIRS = [('CF', 'AF'), ('AF', 'CF')]  # (source: trained on, target: cross-domain test)

DATA_ROOT = './data'
GDRIVE_FILE_ID = '1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf'
ARCHIVE_PATH = os.path.join(DATA_ROOT, 'SCUT-FBP5500_v2.1.zip')
EXTRACT_DIR = os.path.join(DATA_ROOT, 'SCUT-FBP5500_v2')
IMAGES_DIR = os.path.join(EXTRACT_DIR, 'Images')
ALL_LABELS_PATH = os.path.join(EXTRACT_DIR, 'train_test_files', 'All_labels.txt')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

torch.manual_seed(0)

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
    """Returns a list of (filename, raw_score, binary_label) for the given 2-letter race/gender prefix
    (CF/CM/AF/AM), median-split on that subset's own score distribution."""
    records = []
    with open(ALL_LABELS_PATH) as f:
        for line in f:
            filename, score = line.split()
            if filename.startswith(prefix):
                records.append((filename, float(score)))
    scores = np.array([s for _, s in records])
    threshold = np.percentile(scores, PRETTY_PERCENTILE)
    return [(fn, score, int(score > threshold)) for fn, score in records], threshold


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
    metrics = {
        'n': len(labels),
        'accuracy': float(((probs > 0.5) == labels).mean()),
        'roc_auc': float(auc(fpr, tpr)),
        'mean_score_pretty': float(probs[labels == 1].mean()),
        'mean_score_average': float(probs[labels == 0].mean()),
    }
    return probs, labels, fpr, tpr, metrics


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


ensure_dataset()

direction_results = []
for source, target in TRANSFER_PAIRS:
    print(f'\n=== Train on {source}, evaluate in-domain ({source}) + cross-domain ({target}) ===')
    source_items, source_threshold = build_items(source)
    target_items, target_threshold = build_items(target)
    source_labels = np.array([label for _, _, label in source_items])
    print(f'{source}: n={len(source_items)}  score_threshold={source_threshold:.3f}   '
          f'{target}: n={len(target_items)}  score_threshold={target_threshold:.3f}')

    train_items, temp_items = train_test_split(
        source_items, train_size=TRAIN_FRAC, stratify=source_labels, random_state=0)
    temp_labels = np.array([label for _, _, label in temp_items])
    val_items, test_items = train_test_split(
        temp_items, train_size=VAL_FRAC / (VAL_FRAC + TEST_FRAC), stratify=temp_labels, random_state=0)
    print(f'{source} train={len(train_items)}  val={len(val_items)}  in-domain test={len(test_items)}')

    t0 = time.time()
    model, epochs_run, best_val_loss = train_model(train_items, val_items)
    print(f'Trained {epochs_run} epochs, best val loss {best_val_loss:.4f}  ({time.time() - t0:.1f}s)')

    in_probs, in_labels, in_fpr, in_tpr, in_metrics = evaluate(model, test_items)
    cross_probs, cross_labels, cross_fpr, cross_tpr, cross_metrics = evaluate(model, target_items)

    print(f'In-domain ({source} held-out, n={in_metrics["n"]}):    '
          f'accuracy={in_metrics["accuracy"]:.3f}  roc_auc={in_metrics["roc_auc"]:.3f}')
    print(f'Cross-domain ({target} full, n={cross_metrics["n"]}):  '
          f'accuracy={cross_metrics["accuracy"]:.3f}  roc_auc={cross_metrics["roc_auc"]:.3f}')

    direction_results.append({
        'source': source, 'target': target,
        'source_score_threshold': float(source_threshold), 'target_score_threshold': float(target_threshold),
        'n_train': len(train_items), 'n_val': len(val_items),
        'epochs_run': epochs_run, 'best_val_loss': float(best_val_loss),
        'in_domain': in_metrics, 'cross_domain': cross_metrics,
        'in_domain_fpr': in_fpr.tolist(), 'in_domain_tpr': in_tpr.tolist(),
        'cross_domain_fpr': cross_fpr.tolist(), 'cross_domain_tpr': cross_tpr.tolist(),
        'model': model,
        'target_items': target_items, 'cross_probs': cross_probs, 'cross_labels': cross_labels,
    })

# --- Visualization: per direction, in-domain vs cross-domain ROC + Grad-CAM on cross-domain extremes ---
n_dirs = len(direction_results)
fig = plt.figure(figsize=(11, 6.5 * n_dirs))
gs = fig.add_gridspec(2 * n_dirs, 2 * N_EXAMPLES, height_ratios=[1.3, 1] * n_dirs)

for d, res in enumerate(direction_results):
    roc_row, cam_row = 2 * d, 2 * d + 1

    ax_roc = fig.add_subplot(gs[roc_row, :])
    ax_roc.plot(res['in_domain_fpr'], res['in_domain_tpr'], color='tab:blue',
                label=f"In-domain {res['source']} (AUC={res['in_domain']['roc_auc']:.3f})")
    ax_roc.plot(res['cross_domain_fpr'], res['cross_domain_tpr'], color='tab:red',
                label=f"Cross-domain {res['target']} (AUC={res['cross_domain']['roc_auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=0.8, label='Chance')
    ax_roc.set_xlabel('False positive rate')
    ax_roc.set_ylabel('True positive rate')
    ax_roc.set_title(f"Trained on {res['source']}: in-domain vs. cross-domain ({res['target']}) ROC")
    ax_roc.legend(loc='lower right', fontsize=8)

    cross_probs, cross_labels = res['cross_probs'], res['cross_labels']
    target_filenames = [fn for fn, _, _ in res['target_items']]
    top_idx = np.argsort(-cross_probs)[:N_EXAMPLES]
    bottom_idx = np.argsort(cross_probs)[:N_EXAMPLES]
    gradcam_target_layer = res['model'].layer4[-1]

    for col, idx in enumerate(list(top_idx) + list(bottom_idx)):
        img = Image.open(os.path.join(IMAGES_DIR, target_filenames[idx])).convert('RGB')
        face = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        cam = gradcam_heatmap(res['model'], gradcam_target_layer, eval_transform(img))
        true_label = 'pretty' if cross_labels[idx] == 1 else 'average'

        ax = fig.add_subplot(gs[cam_row, col])
        ax.imshow(face)
        ax.imshow(cam, cmap='jet', alpha=0.45)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{cross_probs[idx]:.2f} ({true_label})', fontsize=8)
        if col == 0:
            ax.set_ylabel(f"{res['target']} examples", fontsize=9, rotation=0, ha='right', va='center')

fig.tight_layout()
os.makedirs('results', exist_ok=True)
fig.savefig('results/scut_fbp_beauty_cross_race_transfer.png', dpi=150)
print('\nSaved visualization to results/scut_fbp_beauty_cross_race_transfer.png')

metrics = {
    'pretty_percentile': PRETTY_PERCENTILE,
    'min_epochs': MIN_EPOCHS,
    'max_epochs': MAX_EPOCHS,
    'patience': PATIENCE,
    'directions': [
        {k: v for k, v in r.items()
         if k not in ('model', 'target_items', 'cross_probs', 'cross_labels', 'in_domain_fpr', 'in_domain_tpr',
                      'cross_domain_fpr', 'cross_domain_tpr')}
        for r in direction_results
    ],
}
with open('results/scut_fbp_beauty_cross_race_transfer_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
log_experiment('scut_fbp_beauty_cross_race_transfer', metrics)
