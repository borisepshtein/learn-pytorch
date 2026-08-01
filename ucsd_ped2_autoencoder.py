import json
import os
import tarfile
import time
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from experiment_log import log_experiment

IMG_SIZE = 128
LATENT_DIM = 128
BATCH_SIZE = 32
N_EXAMPLES = 5  # per row in the qualitative example grid

MIN_EPOCHS = 30
MAX_EPOCHS = 100
PATIENCE = 5  # epochs to wait for a significant val-loss improvement before stopping
MIN_REL_IMPROVEMENT = 0.01  # val loss must drop by at least 1% to count as improvement
VAL_FRACTION = 0.1  # held-out slice of the normal training frames, used only for early stopping

DATA_ROOT = './data'
DATASET_URL = 'http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz'
ARCHIVE_PATH = os.path.join(DATA_ROOT, 'UCSD_Anomaly_Dataset.tar.gz')
EXTRACT_DIR = os.path.join(DATA_ROOT, 'UCSD_Anomaly_Dataset.v1p2')
PED2_DIR = os.path.join(EXTRACT_DIR, 'UCSDped2')

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


def ensure_dataset():
    """Download + extract the UCSD Anomaly Dataset (both Ped1 and Ped2 ship in one archive) if not already present."""
    os.makedirs(DATA_ROOT, exist_ok=True)
    if os.path.isdir(PED2_DIR):
        return
    if not os.path.isfile(ARCHIVE_PATH):
        print(f'Downloading UCSD Anomaly Dataset from {DATASET_URL} (includes Ped1 + Ped2, ~a few hundred MB) ...')
        try:
            urllib.request.urlretrieve(DATASET_URL, ARCHIVE_PATH)
        except Exception as e:
            raise RuntimeError(
                f'Automatic download failed ({e}).\n'
                f'Please download it manually from http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm '
                f'and extract it so that {PED2_DIR} exists.'
            ) from e
    print(f'Extracting {ARCHIVE_PATH} ...')
    with tarfile.open(ARCHIVE_PATH) as tar:
        tar.extractall(DATA_ROOT)
    if not os.path.isdir(PED2_DIR):
        raise RuntimeError(f'Extraction finished but {PED2_DIR} was not found; the dataset layout may have changed.')


FRAME_EXTS = {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'}


def list_frames(dir_path):
    if not os.path.isdir(dir_path):
        return []
    names = [f for f in os.listdir(dir_path) if os.path.splitext(f)[1].lower() in FRAME_EXTS]
    return [os.path.join(dir_path, f) for f in sorted(names)]


def is_clip_dir(path):
    return os.path.isdir(path) and not os.path.basename(path).endswith('_gt')


def is_readable_image(path):
    """Some frames in the archive come through corrupted/truncated (flaky old download server, no checksum)."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_train_frames():
    clip_dirs = sorted(p for p in
                        (os.path.join(PED2_DIR, 'Train', d) for d in os.listdir(os.path.join(PED2_DIR, 'Train')))
                        if is_clip_dir(p))
    paths = []
    for clip_dir in clip_dirs:
        paths.extend(list_frames(clip_dir))
    valid_paths = [p for p in paths if is_readable_image(p)]
    n_skipped = len(paths) - len(valid_paths)
    if n_skipped:
        print(f'Skipped {n_skipped} unreadable/corrupt training frame(s)')
    return valid_paths


def collect_test_frames():
    """Returns (frame_paths, labels) where labels[i]=1 if that frame's ground-truth mask has any anomaly pixel."""
    clip_dirs = sorted(p for p in
                        (os.path.join(PED2_DIR, 'Test', d) for d in os.listdir(os.path.join(PED2_DIR, 'Test')))
                        if is_clip_dir(p))
    paths, labels = [], []
    n_skipped = 0
    for clip_dir in clip_dirs:
        frames = list_frames(clip_dir)
        gt_frames = list_frames(clip_dir + '_gt')
        for i, frame_path in enumerate(frames):
            if not is_readable_image(frame_path):
                n_skipped += 1
                continue
            label = 0
            if i < len(gt_frames) and is_readable_image(gt_frames[i]):
                mask = np.array(Image.open(gt_frames[i]))
                label = int(mask.any())
            paths.append(frame_path)
            labels.append(label)
    if n_skipped:
        print(f'Skipped {n_skipped} unreadable/corrupt test frame(s)')
    return paths, labels


frame_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


class FrameDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        return frame_transform(img)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
        )
        self.to_latent = nn.Linear(128 * 16 * 16, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 128 * 16 * 16)
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    # 64 -> 128
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv_encoder(x)
        b = feat.shape[0]
        latent = torch.relu(self.to_latent(feat.flatten(1)))
        feat = torch.relu(self.from_latent(latent)).view(b, 128, 16, 16)
        return self.conv_decoder(feat)


ensure_dataset()

train_paths = collect_train_frames()
test_paths, test_labels = collect_test_frames()
test_labels = np.array(test_labels)

print(f'Train frames (normal only): {len(train_paths)}')
print(f'Test frames: {len(test_paths)}  (normal={int((test_labels == 0).sum())}, anomalous={int((test_labels == 1).sum())})')

val_rng = np.random.default_rng(0)
perm = val_rng.permutation(len(train_paths))
n_val = max(1, int(len(train_paths) * VAL_FRACTION))
val_paths = [train_paths[i] for i in perm[:n_val]]
fit_paths = [train_paths[i] for i in perm[n_val:]]
print(f'Training on {len(fit_paths)} frames, holding out {len(val_paths)} for early-stopping validation')

train_loader = DataLoader(FrameDataset(fit_paths), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(FrameDataset(val_paths), batch_size=BATCH_SIZE, shuffle=False)

model = ConvAutoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

best_val_loss = float('inf')
best_state = None
epochs_without_improvement = 0
epoch = 0

t0 = time.time()
while epoch < MAX_EPOCHS:
    epoch += 1
    model.train()
    running_loss = 0.0
    t_epoch = time.time()
    for frames in train_loader:
        frames = frames.to(device)
        optimizer.zero_grad()
        recon = model(frames)
        loss = criterion(recon, frames)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for frames in val_loader:
            frames = frames.to(device)
            recon = model(frames)
            val_running_loss += criterion(recon, frames).item()
    val_loss = val_running_loss / len(val_loader)

    print(f'Epoch {epoch}/{MAX_EPOCHS}  train loss: {train_loss:.5f}  val loss: {val_loss:.5f}  ({time.time() - t_epoch:.1f}s)')

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
print(f'Total training time: {time.time() - t0:.1f}s  ({epoch} epochs run, best val loss {best_val_loss:.5f})')

# --- Evaluation: per-frame reconstruction error as an anomaly score ---
test_loader = DataLoader(FrameDataset(test_paths), batch_size=BATCH_SIZE, shuffle=False)

model.eval()
all_frames, all_recon, all_mse = [], [], []
with torch.no_grad():
    for frames in test_loader:
        frames = frames.to(device)
        recon = model(frames)
        mse = ((recon - frames) ** 2).flatten(1).mean(dim=1)
        all_frames.append(frames.cpu())
        all_recon.append(recon.cpu())
        all_mse.append(mse.cpu())

all_frames = torch.cat(all_frames)
all_recon = torch.cat(all_recon)
all_mse = torch.cat(all_mse).numpy()

fpr, tpr, _ = roc_curve(test_labels, all_mse)
roc_auc = auc(fpr, tpr)

mean_normal = all_mse[test_labels == 0].mean()
mean_abnormal = all_mse[test_labels == 1].mean()
print(f'\nMean reconstruction MSE  normal={mean_normal:.5f}  anomalous={mean_abnormal:.5f}')
print(f'ROC AUC (frame-level anomaly detection): {roc_auc:.3f}')

# --- Visualization: error-over-time, example reconstructions, ROC curve ---
fig = plt.figure(figsize=(14, 14))
gs = fig.add_gridspec(4, N_EXAMPLES, height_ratios=[1.2, 1, 1, 1.4])

ax_trace = fig.add_subplot(gs[0, :])
ax_trace.plot(all_mse, color='tab:blue', linewidth=0.8, label='Reconstruction MSE')
ax_trace.fill_between(np.arange(len(all_mse)), 0, all_mse.max(), where=test_labels == 1,
                       color='tab:red', alpha=0.15, label='Ground-truth anomalous frame')
ax_trace.set_xlabel('Test frame index')
ax_trace.set_ylabel('Reconstruction MSE')
ax_trace.set_title('Per-frame reconstruction error across the full test set')
ax_trace.legend(loc='upper right')

normal_idx = np.where(test_labels == 0)[0]
abnormal_idx = np.where(test_labels == 1)[0]
example_normal = np.random.default_rng(0).choice(normal_idx, size=min(N_EXAMPLES, len(normal_idx)), replace=False)
example_abnormal = np.random.default_rng(0).choice(abnormal_idx, size=min(N_EXAMPLES, len(abnormal_idx)), replace=False)

for row, (label, idx) in enumerate([('Normal — input / recon', example_normal),
                                     ('Anomalous — input / recon', example_abnormal)]):
    for col in range(len(idx)):
        ax = fig.add_subplot(gs[row + 1, col])
        top = all_frames[idx[col], 0]
        bottom = all_recon[idx[col], 0]
        combined = torch.cat([top, bottom], dim=0)
        ax.imshow(combined, cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(label, fontsize=9, rotation=0, ha='right', va='center')

ax_roc = fig.add_subplot(gs[3, :])
ax_roc.plot(fpr, tpr, color='tab:blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=0.8, label='Chance')
ax_roc.set_xlabel('False positive rate')
ax_roc.set_ylabel('True positive rate')
ax_roc.set_title('Frame-level anomaly detection ROC (UCSD Ped2)')
ax_roc.legend(loc='lower right')

fig.tight_layout()
os.makedirs('results', exist_ok=True)
fig.savefig('results/ucsd_ped2_anomaly.png', dpi=150)
print('\nSaved visualization to results/ucsd_ped2_anomaly.png')

metrics = {
    'img_size': IMG_SIZE,
    'latent_dim': LATENT_DIM,
    'min_epochs': MIN_EPOCHS,
    'max_epochs': MAX_EPOCHS,
    'patience': PATIENCE,
    'epochs_run': epoch,
    'best_val_loss': float(best_val_loss),
    'n_train_frames': len(fit_paths),
    'n_val_frames': len(val_paths),
    'n_test_frames': len(test_paths),
    'n_test_normal': int((test_labels == 0).sum()),
    'n_test_anomalous': int((test_labels == 1).sum()),
    'mean_mse_normal': float(mean_normal),
    'mean_mse_anomalous': float(mean_abnormal),
    'roc_auc': float(roc_auc),
}
with open('results/ucsd_ped2_anomaly_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
log_experiment('ucsd_ped2_autoencoder', metrics)
