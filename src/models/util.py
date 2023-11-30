import torch


class Unsqueeze(torch.nn.Module):
    def __init__(self, dim: int = 0, n_dims: int = 1):
        super(Unsqueeze, self).__init__()
        self.dim = dim
        self.n_dims = n_dims

    def forward(self, x):
        for i in range(self.n_dims):
            x = torch.unsqueeze(x, self.dim)
        return x


if __name__ == "__main__":
    dummy_input = torch.zeros(1, 1024)
    unsqueeze = Unsqueeze(dim=2, n_dims=2)

    dummy_input = unsqueeze(dummy_input)
    print(dummy_input.shape)



