import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


model = MnistNet()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

EPOCHS = 5

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch}/{EPOCHS}  avg loss: {running_loss / len(train_loader):.4f}')

model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        preds = model(data).argmax(dim=1)
        correct += preds.eq(target).sum().item()

total = len(test_data)
error = 100.0 * (total - correct) / total
print(f'\nTest set: {correct}/{total} correct  |  error rate: {error:.2f}%')

print('\nSample predictions (first 10 test images):')
sample_data, sample_target = next(iter(DataLoader(test_data, batch_size=10)))
with torch.no_grad():
    sample_preds = model(sample_data).argmax(dim=1)
for i, (pred, true) in enumerate(zip(sample_preds, sample_target)):
    status = 'OK' if pred == true else 'WRONG'
    print(f'  {i+1:2d}.  predicted={pred.item()}  actual={true.item()}  {status}')
