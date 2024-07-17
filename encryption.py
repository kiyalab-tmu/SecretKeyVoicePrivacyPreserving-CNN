import torch
from torch import nn
from secret_key import SecretKey

class EncryptedConv1d(nn.Module):
    def __init__(self, conv1d: nn.Conv1d, secret_key: SecretKey) -> None:
        super().__init__()

        assert conv1d.kernel_size == conv1d.stride, "Kernel size and stride size must be the same !!"

        self.encrypted_conv1d = nn.Conv1d(
            conv1d.in_channels, 
            conv1d.out_channels, 
            conv1d.kernel_size, 
            conv1d.stride, 
            conv1d.padding, 
            conv1d.dilation, 
            conv1d.groups, 
            False if conv1d.bias is None else True, 
            conv1d.padding_mode)

        with torch.no_grad():
            for i in range(self.encrypted_conv1d.out_channels):
                self.encrypted_conv1d.weight[i][0] = torch.matmul(secret_key.kernel_key, conv1d.weight[i][0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encrypted_conv1d(x)

class EncryptedConv2d(nn.Module):
    pass

class Cipher:
    def __init__(self, secret_key: SecretKey) -> None:
        self.secret_key = secret_key

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x.view(x.shape[1]//self.secret_key.key_dims, self.secret_key.key_dims)
        x2 = torch.matmul(x1, self.secret_key.data_key)
        y = x2.view(1, x.shape[1])
        return y