# Speech privacy-preserving methods using secret key for convolutional neural network models and their robustness evaluation


# Abstract
In this paper, we propose privacy-preserving methods with a secret key for convolutional neural network (CNN)-based models in speech processing tasks. 
In environments where untrusted third parties, like cloud servers, provide CNN-based systems, ensuring the privacy of speech queries becomes essential. 
This paper proposes encryption methods for speech queries using secret keys and a model structure that allows for encrypted queries to be accepted without decryption.
Our approach introduces three types of secret keys: Shuffling, Flipping, and random orthogonal matrix (ROM).
In experiments, we demonstrate that when the proposed methods are used with the correct key, identification performance did not degrade. 
Conversely, when an incorrect key is used, the performance significantly decreased. 
Particularly, with the use of ROM, we show that even with a relatively small key space, high privacy-preserving performance can be maintained many speech processing tasks. Furthermore, we also demonstrate the difficulty of recovering original speech from encrypted queries in various robustness evaluations.

## Examples
### 1. Encryption of audio signals (when using pytorch's Dataset & Dataloader)
```python
from encryption import Cipher

class AudioDataset(data.Dataset):
    def __init__(self, wav_files: List[str], num_utt_samples: int=59049, cipher: Cipher=None) -> None:
        self.num_utt_samples = num_utt_samples
        self.wav_files = wav_files
        self.cipher = cipher
    
    def __len__(self) -> int:
        return len(self.wav_files)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        x, sf = torchaudio.load(self.wav_files[index])
        x = x[:, : self.num_utt_samples]
        x = self.cipher(x)
        return x, self.wav_files[index]
```

### 2. Encryption of Audio model
Replace the first convolution layer of the model with an encrypted convolution layer.

```python
# when using rawnext model (https://github.com/wngh1187/RawNeXt)
from secret_key import SecretKey
from encryption import EncryptedConv1d
from torch import nn

def encrypt_rawnext_model(rawnext_model: nn.Module, secret_key: SecretKey) -> nn.Module:
    model = copy.deepcopy(rawnext_model)
    model.base_layer[0] = EncryptedConv1d(conv1d=model.base_layer[0], secret_key=secret_key)
    return model
```
### 3. Key generation
Call generate_keys() function with specified parameters.
```python
from secret_key import SecretKey, KeyType
from secret_key_generator import generate_keys

key_type = KeyType.ROM
num_keys = 10
key_dims = 3
num_trials = -1
key_prefix = "rom"
key_dir = "./keys"
generate_keys(key_type, num_keys, key_dims, num_trials, key_prefix, key_dir)    
```

# Citation
