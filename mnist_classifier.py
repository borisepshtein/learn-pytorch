import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

TARGET_DIGIT = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_data = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=1000)


class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3),   # 28x28 -> 26x26
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),  # 26x26 -> 24x24
            nn.ReLU(),
            nn.MaxPool2d(2),       # 24x24 -> 12x12
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


model = MnistNet().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

EPOCHS = 5

t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    t_epoch = time.time()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        binary_target = (target == TARGET_DIGIT).float()
        optimizer.zero_grad()
        loss = criterion(model(data), binary_target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch}/{EPOCHS}  avg loss: {running_loss / len(train_loader):.4f}  ({time.time() - t_epoch:.1f}s)')
print(f'Total training time: {time.time() - t0:.1f}s')

model.eval()
fp = fn = 0
total_pos = total_neg = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        binary_target = (target == TARGET_DIGIT)
        preds = model(data) > 0  # logit > 0 means probability > 0.5

        total_pos += binary_target.sum().item()
        total_neg += (~binary_target).sum().item()
        fp += (preds & ~binary_target).sum().item()
        fn += (~preds & binary_target).sum().item()

print(f'\nDigit "{TARGET_DIGIT}" vs. rest  |  test set: {total_pos} positives, {total_neg} negatives')
print(f'False Positives: {fp}/{total_neg} = {100.0 * fp / total_neg:.2f}%  (predicted 3, was not 3)')
print(f'False Negatives: {fn}/{total_pos} = {100.0 * fn / total_pos:.2f}%  (missed a 3)')
