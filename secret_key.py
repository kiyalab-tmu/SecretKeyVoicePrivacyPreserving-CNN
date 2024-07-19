from typing import Tuple
from enum import Enum
from scipy.stats import ortho_group
import torch
import numpy as np
import pickle

rng = np.random.default_rng()

class KeyType(Enum):
	ROM=0,
	SHUFFLE=1,
	FLIP=2

class SecretKey:
	def __init__(self, key_type:KeyType = KeyType.ROM, key_dims:int = 3, key_file:str = None) -> None:
		self.__data_key: torch.Tensor = None
		self.__kernel_key: torch.Tensor = None
		self.__key_type: KeyType = key_type
		self.__key_dims: int = key_dims

		if key_file is not None:
			self.load(key_file)
		else:
			if self.__key_type == KeyType.ROM:
				self.__data_key, self.__kernel_key = self.__gen_rom_key(self.__key_dims)
			elif self.__key_type == KeyType.SHUFFLE:
				self.__data_key, self.__kernel_key = self.__gen_shuffle_key(self.__key_dims)
			elif self.__key_type == KeyType.FLIP:
				self.__data_key, self.__kernel_key = self.__gen_flip_key(self.__key_dims)
					
	def load(self, key_file:str):
		with open(key_file, "rb") as f:
			self.__dict__.update(**pickle.load(f))

	def save(self, key_file:str):
		with open(key_file, "wb") as f:
			pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

	@property
	def data_key(self) -> torch.Tensor:
		return self.__data_key
	
	@property
	def kernel_key(self) -> torch.Tensor:
		return self.__kernel_key
	
	@property
	def key_type(self) -> KeyType:
		return self.__key_type
	
	@property
	def key_dims(self) -> int:
		return self.__key_dims

	def __gen_rom_key(self, ndim: int) -> Tuple[torch.Tensor, torch.Tensor]:
		key_matrix = torch.Tensor(ortho_group.rvs(ndim))
		return key_matrix, key_matrix.T

	def __gen_shuffle_key(self, ndim: int) -> Tuple[torch.Tensor, torch.Tensor]:
		idx = np.arange(ndim)
		rng.shuffle(idx)
		key_matrix = torch.Tensor(np.eye(ndim)[:,idx])
		return key_matrix, torch.Tensor(np.linalg.inv(key_matrix))

	def __gen_flip_key(self, ndim: int) -> Tuple[torch.Tensor, torch.Tensor]:
		key_matrix = torch.Tensor(np.diag(np.sign(rng.uniform(-1, 1, (ndim)))))
		return key_matrix, torch.Tensor(np.linalg.inv(key_matrix))
