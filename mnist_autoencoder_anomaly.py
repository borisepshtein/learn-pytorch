import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

TRAIN_DIGIT = 3
ANOMALY_DIGIT = 7
EPOCHS = 30
N_EXAMPLES = 6
N_EVAL_REPEATS = 5  # avg per-image error over multiple corruption draws to cut evaluation noise
LATENT_DIM = 16  # tighter bottleneck -> stronger specialization to the training digit

ELASTIC_ALPHA_RANGE = (10, 18)
ELASTIC_SIGMA_RANGE = (4, 7)
BLUR_SIGMA_RANGE = (0.6, 1.6)
NOISE_STD_RANGE = (0.15, 0.30)

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

to_tensor = transforms.ToTensor()

train_data = datasets.MNIST('./data', train=True, download=True, transform=to_tensor)
test_data = datasets.MNIST('./data', train=False, download=True, transform=to_tensor)

train_digit_idx = (train_data.targets == TRAIN_DIGIT).nonzero(as_tuple=True)[0]
train_subset = Subset(train_data, train_digit_idx)
train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)

test_digit_idx = (test_data.targets == TRAIN_DIGIT).nonzero(as_tuple=True)[0]
test_anomaly_idx = (test_data.targets == ANOMALY_DIGIT).nonzero(as_tuple=True)[0]


def elastic_warp_one(img):
    """Random elastic (nonlinear) warp for a single image. img: (1,1,H,W) in [0,1]."""
    _, _, h, w = img.shape
    alpha = torch.empty(1).uniform_(*ELASTIC_ALPHA_RANGE).item()
    sigma = torch.empty(1).uniform_(*ELASTIC_SIGMA_RANGE).item()

    dx = torch.rand(1, 1, h, w, device=img.device) * 2 - 1
    dy = torch.rand(1, 1, h, w, device=img.device) * 2 - 1

    ksize = int(sigma * 4) | 1  # force odd
    dx = transforms.functional.gaussian_blur(dx, kernel_size=ksize, sigma=sigma) * alpha
    dy = transforms.functional.gaussian_blur(dy, kernel_size=ksize, sigma=sigma) * alpha

    yy, xx = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=torch.float32),
        torch.arange(w, device=img.device, dtype=torch.float32),
        indexing='ij',
    )
    grid_x = 2 * (xx + dx.squeeze(0).squeeze(0)) / (w - 1) - 1
    grid_y = 2 * (yy + dy.squeeze(0).squeeze(0)) / (h - 1) - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    return F.grid_sample(img, grid, align_corners=True, padding_mode='reflection')


def corrupt_one(img):
    """Elastic warp -> blur -> noise for a single image, each with its own random severity."""
    img = elastic_warp_one(img)

    blur_sigma = torch.empty(1).uniform_(*BLUR_SIGMA_RANGE).item()
    img = transforms.functional.gaussian_blur(img, kernel_size=3, sigma=blur_sigma)

    noise_std = torch.empty(1).uniform_(*NOISE_STD_RANGE).item()
    img = img + torch.randn_like(img) * noise_std
    return img.clamp(0, 1)


def corrupt(images):
    """Apply corrupt_one independently to every image in the batch."""
    return torch.cat([corrupt_one(images[i:i + 1]) for i in range(images.shape[0])], dim=0)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
        )
        # Dense bottleneck compresses 64*7*7=3136 conv features down to
        # latent_dim. Without this, the conv feature map (already bigger
        # than the 784-pixel input) isn't an actual bottleneck, so the model
        # learns a generic deblur/denoise filter instead of specializing to
        # the training digit. Extra conv capacity above lets it fit digit-3
        # more precisely without loosening this bottleneck.
        self.to_latent = nn.Linear(64 * 7 * 7, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 64 * 7 * 7)
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv_encoder(x)
        b = feat.shape[0]
        latent = F.relu(self.to_latent(feat.flatten(1)))
        feat = F.relu(self.from_latent(latent)).view(b, 64, 7, 7)
        return self.conv_decoder(feat)


model = ConvAutoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    t_epoch = time.time()
    for clean, _ in train_loader:
        clean = clean.to(device)
        noisy = corrupt(clean)

        optimizer.zero_grad()
        recon = model(noisy)
        loss = criterion(recon, clean)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch}/{EPOCHS}  avg loss: {running_loss / len(train_loader):.4f}  ({time.time() - t_epoch:.1f}s)')
print(f'Total training time: {time.time() - t0:.1f}s')


def per_image_error(indices):
    """Corrupt, reconstruct, and return (clean, noisy, recon, per-image MSE) for the given test indices.

    MSE is averaged over N_EVAL_REPEATS independent corruption draws per image,
    so the histogram reflects the model's real generalization gap rather than
    noise from a single random corruption draw per image.
    """
    clean = torch.stack([test_data[i][0] for i in indices]).to(device)
    model.eval()
    mse_sum = torch.zeros(clean.shape[0], device=device)
    noisy = recon = None
    with torch.no_grad():
        for _ in range(N_EVAL_REPEATS):
            noisy = corrupt(clean)
            recon = model(noisy)
            mse_sum += ((recon - clean) ** 2).flatten(1).mean(dim=1)
    mse = mse_sum / N_EVAL_REPEATS
    return clean.cpu(), noisy.cpu(), recon.cpu(), mse.cpu()


clean3, noisy3, recon3, mse3 = per_image_error(test_digit_idx)
clean7, noisy7, recon7, mse7 = per_image_error(test_anomaly_idx)

print(f'\nDigit {TRAIN_DIGIT} (trained on)  test restoration MSE: mean={mse3.mean():.4f}  std={mse3.std():.4f}  n={len(mse3)}')
print(f'Digit {ANOMALY_DIGIT} (never seen) test restoration MSE: mean={mse7.mean():.4f}  std={mse7.std():.4f}  n={len(mse7)}')
print(f'Mean error ratio (digit {ANOMALY_DIGIT} / digit {TRAIN_DIGIT}): {mse7.mean() / mse3.mean():.2f}x')

example_idx3 = torch.randperm(len(clean3))[:N_EXAMPLES]
example_idx7 = torch.randperm(len(clean7))[:N_EXAMPLES]

fig = plt.figure(figsize=(2 * N_EXAMPLES, 14))
gs = fig.add_gridspec(7, N_EXAMPLES, height_ratios=[1, 1, 1, 1, 1, 1, 1.6])

row_specs = [
    (f'Digit {TRAIN_DIGIT} — corrupted input', noisy3, example_idx3),
    (f'Digit {TRAIN_DIGIT} — clean original', clean3, example_idx3),
    (f'Digit {TRAIN_DIGIT} — reconstruction', recon3, example_idx3),
    (f'Digit {ANOMALY_DIGIT} — corrupted input', noisy7, example_idx7),
    (f'Digit {ANOMALY_DIGIT} — clean original', clean7, example_idx7),
    (f'Digit {ANOMALY_DIGIT} — reconstruction', recon7, example_idx7),
]

for row, (label, images, idx) in enumerate(row_specs):
    for col in range(N_EXAMPLES):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(images[idx[col], 0], cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(label, fontsize=9, rotation=0, ha='right', va='center')

ax_hist = fig.add_subplot(gs[6, :])
bins = 40
ax_hist.hist(mse3.numpy(), bins=bins, alpha=0.6, label=f'Digit {TRAIN_DIGIT} (trained on)', color='tab:blue')
ax_hist.hist(mse7.numpy(), bins=bins, alpha=0.6, label=f'Digit {ANOMALY_DIGIT} (anomaly)', color='tab:red')
ax_hist.set_xlabel('Per-image reconstruction MSE')
ax_hist.set_ylabel('Count')
ax_hist.set_title('Restoration error distribution: trained digit vs. anomaly digit')
ax_hist.legend()

fig.tight_layout()
fig.savefig('autoencoder_anomaly.png', dpi=150)
print('\nSaved visualization to autoencoder_anomaly.png')
