import time
import torch

from torchvision.transforms import *
from torchvision.datasets import MNIST
from torchvision.models import mobilenet_v2

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

import torch.optim as optim
import torch.nn.functional as f


def test_trainAndtest():
    ts = time.time()
    for epoch in range(3):
        start = time.time()
        train(epoch)
        test()
        end = time.time()
        print('Run time for epoch %d: %f ms' % (epoch, (end - start) * 1000))
    tf = time.time()
    print('Total run time : %f ms' % ((tf - ts) * 1000))


def train(epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # to cuda if available
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        out = net(data)
        if torch.cuda.is_available():
            out = out.cuda()
        loss = f.cross_entropy(out, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"train epoch: {epoch} [{(batch_idx + 1) * len(data)}/{len(train_loader.dataset)}"
                  f" ({100 * (batch_idx + 1) / len(train_loader):.0f}%)] loss: {loss.item():.6f}")


def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # to cuda if available
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            out = net(data)
            if torch.cuda.is_available():
                out = out.cuda()

            test_loss += f.cross_entropy(out, target).item()

            pred = out.data.max(1, keepdim=True)[1]

            correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(test_loader.dataset)
        print(f"\ntrain: loss: {test_loss:.6f}, "
              f"acc: {correct}/{len(test_loader.dataset)} ({100 * correct / len(test_loader.dataset)}%)\n")


def test_BuildAndForward():
    for layer in net.named_modules():
        print(layer)

    net.train()
    input = torch.ones(10, 3, 28, 28)
    out = net(input)
    print(out.shape)
    print(out)


if __name__ == '__main__':
    # init
    net = mobilenet_v2(num_classes=10)
    if torch.cuda.is_available():
        print("cuda available")
        device = torch.device("cuda:0")
        net.to(device)
        torch.cuda.empty_cache()

    # "grow" method in java
    transform = Compose(
        [ToTensor(), Lambda(lambda t: torch.cat([t, t, t], 0))])  # think concat dim to be 1, confused...

    mnist_train = MNIST("MNIST", train=True, transform=transform, download=False)
    mnist_test = MNIST("MNIST", train=False, transform=transform, download=False)
    train_loader = DataLoader(mnist_train, batch_size=64, sampler=SequentialSampler(mnist_train))
    test_loader = DataLoader(mnist_test, batch_size=64, sampler=SequentialSampler(mnist_test))
    optimizer = optim.SGD(net.parameters(), lr=0.004, momentum=0.5)

    # init finish
    test_BuildAndForward()
    # test_trainAndtest()
