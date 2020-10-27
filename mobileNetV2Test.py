from torchvision.transforms import *
from torchvision.datasets import MNIST
from torchvision.models import MobileNetV2
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.optim as optim
import torch.nn.functional as f
import torch


def train():
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        out = net(data)
        loss = f.cross_entropy(out, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:

            print(f"train epoch: {epoch} [{(batch_idx + 1) * len(data)}/{len(train_loader.dataset)}"
                  f" ({100 * (batch_idx + 1) / len(train_loader):.0f}%)] loss: {loss.item():.6f}")
            break

def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            out = net(data)
            print(out.shape)
            test_loss += f.cross_entropy(out, target).item()

            pred = out.data.max(1, keepdim=True)[1]
            print(pred)
            print(target)
            correct += pred.eq(target.data.view_as(pred)).sum()

            print(pred.eq(target.data.view_as(pred)).sum())
            break
        print(len(test_loader.dataset))
        test_loss /= len(test_loader.dataset)
        print(f"\ntrain: loss: {test_loss:.6f}, "
              f"acc: {correct}/{len(test_loader.dataset)} ({100 * correct / len(test_loader.dataset)}%)\n")

def testForward():
    net.train()
    input =torch.ones(10,3,28,28)
    out = net(input)
    print(out)

if __name__ == '__main__':
    net = MobileNetV2(num_classes=10)
    net.to(device="CUDA")
    # testForward()
    trans = Compose([ToTensor(), Lambda(lambda t: torch.cat([t, t, t], 0))])    # think concat dim to be 1, confused...
    mnist_train = MNIST("MNIST", train=True, transform=trans, download=False)
    mnist_test = MNIST("MNIST", train=False, transform=trans, download=False)
    train_loader = DataLoader(mnist_train, batch_size=64, sampler=SequentialSampler(mnist_train))
    test_loader = DataLoader(mnist_test, batch_size=64, sampler=SequentialSampler(mnist_test))
    optimizer = optim.SGD(net.parameters(), lr=0.004, momentum=0.5)

    for epoch in range(1):
        train()
        test()