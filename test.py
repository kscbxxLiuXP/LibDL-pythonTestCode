import time

import torch
from torch import nn


def test_make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def testRound():
    inp = 32
    oup = 16
    expend_ratio = 1
    hidden_dim = int(round(inp * expend_ratio))
    print(hidden_dim)


def testSomeNNFunction():
    a = torch.tensor((1, 5, 3, 2, 6, 4)).reshape((2, 3))
    b = torch.tensor((0, 1))
    pred = a.max(1, keepdim=True)[1]
    correct = 0;
    print(pred)
    print(b.view_as(pred))
    correct += pred.eq(b.data.view_as(pred)).sum()
    print(pred.eq(b.data.view_as(pred)).sum())
    print(correct)


def zeroPadResize(input, newSize):
    m = torch.nn.ZeroPad2d(newSize)
    if torch.cuda.is_available():
        return m(input).cuda()
    else:
        return m(input)


if __name__ == "__main__":
    input = torch.randn((10, 3, 28, 28))
    print(input.resize_(10,3,224,224).shape)
