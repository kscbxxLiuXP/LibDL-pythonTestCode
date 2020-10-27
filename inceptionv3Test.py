from torchvision.transforms import *
from torchvision.datasets import MNIST
from torchvision.models import Inception3

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

import torch.optim as optim
import torch.nn.functional as f
import torch


def train():
    inception.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data.resize_(64,3,299,299)
        out = inception(data)
        loss = f.cross_entropy(out, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"train epoch: {epoch} [{(batch_idx + 1) * len(data)}/{len(train_loader.dataset)}"
                  f" ({100 * (batch_idx + 1) / len(train_loader):.0f}%)] loss: {loss.item():.6f}")


def test():
    inception.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            out = inception(data)
            test_loss += f.cross_entropy(out, target, reduction='sum').item()
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        print(f"\ntrain: loss: {test_loss:.6f}, "
              f"acc: {correct}/{len(test_loader.dataset)} ({100 * correct / len(test_loader.dataset)}%)\n")


if __name__ == '__main__':
    trans = Compose([ToTensor(), Lambda(lambda t: torch.cat([t, t, t], 0))])    # think concat dim to be 1, confused...
    mnist_train = MNIST("MNIST", train=True, transform=trans, download=False)
    mnist_test = MNIST("MNIST", train=False, transform=trans, download=False)
    train_loader = DataLoader(mnist_train, batch_size=64, sampler=SequentialSampler(mnist_train))
    test_loader = DataLoader(mnist_test, batch_size=64, sampler=SequentialSampler(mnist_test))

    inception = Inception3(num_classes=10)
    optimizer = optim.SGD(inception.parameters(), lr=0.004, momentum=0.5)

    for epoch in range(3):
        train()
        test()