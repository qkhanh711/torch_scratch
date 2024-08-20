from model.pl_modules.simple_nn import Net, Net2
from model.nn_modules.simple_nn import Net as Net3, Net2 as Net4
import torch

def test_net():
    n_input = 784
    n_output = 10
    x = torch.randn(1, n_input)
    net = Net(n_input, n_output)
    out = net(x)
    assert out.size() == (1, n_output), f"Bad output shape: {out.size()}"

    net2 = Net2(n_input, n_output)
    out = net2(x)
    assert out.size() == (1, n_output), f"Bad output shape: {out.size()}"

    net3 = Net3(n_input, n_output)
    out = net3(x)
    assert out.size() == (1, n_output), f"Bad output shape: {out.size()}"

    net4 = Net4(n_input, n_output)
    out = net4(x)
    assert out.size() == (1, n_output), f"Bad output shape: {out.size()}"

if __name__ == "__main__":
    test_net()
    print("Passed all tests!")