"""ResNet18 baseline: predict pretty vs. average from face photos (SCUT-FBP5500, Caucasian-female subset).

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
N_EXAMPLES = 5  # per row in the qualitative example grid

MIN_EPOCHS = 8
MAX_EPOCHS = 40
PATIENCE = 5  # epochs to wait for a significant val-loss improvement before stopping
MIN_REL_IMPROVEMENT = 0.01  # val loss must drop by at least 1% to count as improvement
LEARNING_RATE = 1e-4

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.7, 0.15, 0.15

RACE_GENDER_PREFIX = 'CF'  # filenames are named e.g. 'CF437.jpg': CF=Caucasian female, CM/AF/AM for the others
PRETTY_PERCENTILE = 50  # median split within the subset: score above this percentile -> "pretty"

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
    """Download + extract the SCUT-FBP5500 image archive (Google Drive) if not already present."""
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


def is_target_subset(filename):
    return filename.startswith(RACE_GENDER_PREFIX)


def build_items():
    """Returns a list of (filename, raw_score, binary_label) for the target race/gender subset.

    Reads train_test_files/All_labels.txt bundled in the archive ('filename score', scores in [1, 5],
    mean of 60 raters per image) rather than the same-named files at the top of the GitHub repo, which
    use an older filename scheme that doesn't match this archive's Images/ folder.
    """
    records = []
    with open(ALL_LABELS_PATH) as f:
        for line in f:
            filename, score = line.split()
            if is_target_subset(filename):
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


ensure_dataset()
items, score_threshold = build_items()
labels = np.array([label for _, _, label in items])
print(f'Subset: {RACE_GENDER_PREFIX!r}  n={len(items)}  '
      f'pretty={int(labels.sum())}  average={int((labels == 0).sum())}  score_threshold={score_threshold:.3f}')

train_items, temp_items = train_test_split(items, train_size=TRAIN_FRAC, stratify=labels, random_state=0)
temp_labels = np.array([label for _, _, label in temp_items])
val_items, test_items = train_test_split(
    temp_items, train_size=VAL_FRAC / (VAL_FRAC + TEST_FRAC), stratify=temp_labels, random_state=0)
print(f'Train: {len(train_items)}  Val: {len(val_items)}  Test: {len(test_items)}')

train_loader = DataLoader(FaceDataset(train_items, train_transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(FaceDataset(val_items, eval_transform), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(FaceDataset(test_items, eval_transform), batch_size=BATCH_SIZE, shuffle=False)

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

best_val_loss = float('inf')
best_state = None
epochs_without_improvement = 0
epoch = 0
train_loss_history, val_loss_history = [], []

t0 = time.time()
while epoch < MAX_EPOCHS:
    epoch += 1
    model.train()
    running_loss = 0.0
    t_epoch = time.time()
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images).squeeze(1), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            val_running_loss += criterion(model(images).squeeze(1), targets).item()
    val_loss = val_running_loss / len(val_loader)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    print(f'Epoch {epoch}/{MAX_EPOCHS}  train loss: {train_loss:.4f}  val loss: {val_loss:.4f}  '
          f'({time.time() - t_epoch:.1f}s)')

    if val_loss < best_val_loss * (1 - MIN_REL_IMPROVEMENT):
        best_val_loss = val_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch >= MIN_EPOCHS and epochs_without_improvement >= PATIENCE:
        print(f'Early stopping at epoch {epoch}: val loss has not improved by >={MIN_REL_IMPROVEMENT:.0%} '
              f'for {PATIENCE} epochs')
        break

model.load_state_dict(best_state)
print(f'Total training time: {time.time() - t0:.1f}s  ({epoch} epochs run, best val loss {best_val_loss:.4f})')

# --- Evaluation: sigmoid confidence doubles as the "beauty score" ---
model.eval()
test_probs, test_labels = [], []
with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        probs = torch.sigmoid(model(images).squeeze(1))
        test_probs.append(probs.cpu())
        test_labels.append(targets)
test_probs = torch.cat(test_probs).numpy()
test_labels = torch.cat(test_labels).numpy().astype(int)
test_preds = test_probs > 0.5

accuracy = float((test_preds == test_labels).mean())
total_pos = int(test_labels.sum())
total_neg = int((test_labels == 0).sum())
fp = int((test_preds & (test_labels == 0)).sum())
fn = int((~test_preds & (test_labels == 1)).sum())

fpr, tpr, _ = roc_curve(test_labels, test_probs)
roc_auc = auc(fpr, tpr)

mean_score_pretty = float(test_probs[test_labels == 1].mean())
mean_score_average = float(test_probs[test_labels == 0].mean())

print(f'\nTest accuracy: {accuracy:.3f}  (n={len(test_labels)}, positives={total_pos}, negatives={total_neg})')
print(f'False Positives: {fp}/{total_neg}   False Negatives: {fn}/{total_pos}')
print(f'Mean predicted beauty score  pretty={mean_score_pretty:.3f}  average={mean_score_average:.3f}')
print(f'ROC AUC: {roc_auc:.3f}')

# --- Visualization: loss curves, example faces by predicted score, ROC curve ---
test_filenames = [fn for fn, _, _ in test_items]

fig = plt.figure(figsize=(14, 13))
gs = fig.add_gridspec(4, N_EXAMPLES, height_ratios=[1, 1, 1, 1.3])

ax_loss = fig.add_subplot(gs[0, :])
ax_loss.plot(range(1, epoch + 1), train_loss_history, label='Train loss')
ax_loss.plot(range(1, epoch + 1), val_loss_history, label='Val loss')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('BCE loss')
ax_loss.set_title('Training curve')
ax_loss.legend(loc='upper right')

top_idx = np.argsort(-test_probs)[:N_EXAMPLES]
bottom_idx = np.argsort(test_probs)[:N_EXAMPLES]

for row, (title, idx_list) in enumerate([('Highest predicted score', top_idx), ('Lowest predicted score', bottom_idx)]):
    for col, idx in enumerate(idx_list):
        ax = fig.add_subplot(gs[row + 1, col])
        img = Image.open(os.path.join(IMAGES_DIR, test_filenames[idx])).convert('RGB')
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        true_label = 'pretty' if test_labels[idx] == 1 else 'average'
        ax.set_title(f'{test_probs[idx]:.2f} ({true_label})', fontsize=8)
        if col == 0:
            ax.set_ylabel(title, fontsize=9, rotation=0, ha='right', va='center')

ax_roc = fig.add_subplot(gs[3, :])
ax_roc.plot(fpr, tpr, color='tab:blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=0.8, label='Chance')
ax_roc.set_xlabel('False positive rate')
ax_roc.set_ylabel('True positive rate')
ax_roc.set_title('Pretty vs. average classification ROC (SCUT-FBP5500, Caucasian female)')
ax_roc.legend(loc='lower right')

fig.tight_layout()
os.makedirs('results', exist_ok=True)
fig.savefig('results/scut_fbp_beauty.png', dpi=150)
print('\nSaved visualization to results/scut_fbp_beauty.png')

metrics = {
    'img_size': IMG_SIZE,
    'race_gender_subset': RACE_GENDER_PREFIX,
    'pretty_percentile': PRETTY_PERCENTILE,
    'score_threshold': float(score_threshold),
    'min_epochs': MIN_EPOCHS,
    'max_epochs': MAX_EPOCHS,
    'patience': PATIENCE,
    'epochs_run': epoch,
    'best_val_loss': float(best_val_loss),
    'n_train': len(train_items),
    'n_val': len(val_items),
    'n_test': len(test_items),
    'test_accuracy': accuracy,
    'test_positives': total_pos,
    'test_negatives': total_neg,
    'false_positives': fp,
    'false_negatives': fn,
    'mean_score_pretty': mean_score_pretty,
    'mean_score_average': mean_score_average,
    'roc_auc': float(roc_auc),
}
with open('results/scut_fbp_beauty_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
log_experiment('scut_fbp_beauty_classifier', metrics)
