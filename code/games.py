import torch 
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np 
import random
import json
import os

from PIL import Image


class MujocoGame(Dataset):
	def __init__(self, data_dir, save_dir=None, regenerate=True, eval=False):
		super().__init__()

		self.data_dir = data_dir
		self.save_dir = save_dir
		self.regenerate = regenerate
		self.ids_to_imgs = {}
		self.games = []
		self.eval = eval

		self.create()

	def create(self):
		if self.eval:
			fname = "mujoco/images_test.npy"
		else:
			fname = "mujoco/images_train.npy"
		
		print("\n----------------------------------------")
		print("loading data from {:s}".format(fname))
		print("----------------------------------------\n")

		with open(os.path.join(self.data_dir, fname), "rb") as f:
			imgs = np.load(f)

		means = list(np.mean(imgs, axis=(0, 1, 2)))
		stds = list(np.std(imgs, axis=(0, 1, 2)))

        
		transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize((64, 64), interpolation=Image.NEAREST),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=means,
											                    std=stds)
										])

		for id_, img in enumerate(imgs):
			img = torch.Tensor(np.transpose(img, (2, 0, 1)))
			transformed_img = transform(img)
			self.ids_to_imgs[id_] = transformed_img

	def create_games(self, num_games, num_distractors):
		self.games = []
		for i in range(num_games):
			img_ids = np.random.choice(len(self.ids_to_imgs.items()), num_distractors + 1)
			target_img_idx = np.random.choice(num_distractors + 1)

			game_imgs = torch.stack([self.ids_to_imgs[id_] for id_ in img_ids])
			target_img = game_imgs[target_img_idx]

			self.games.append({"imgs": game_imgs,
							   "labels": target_img_idx,
							   "target_img": target_img})
			
	def __len__(self):
		return len(self.games)

	def __getitem__(self, idx):
		return self.games[idx]


class ShapeGame(Dataset):
	def __init__(self, imgs, labels, num_games=1, num_distractors=4, num_test=None, eval=False):
		super().__init__()

		self._FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
		self._NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15}
		self.num_games = num_games
		self.num_distractors = num_distractors
		self.imgs = imgs
		self.labels = labels
		self.games = []
		self.eval = eval

		if num_test is None:
			self.create_games()
		else:
			self.num_test = num_test

	def create_games(self):
		for i in range(self.num_games):
			img_ids = np.random.choice(len(self.imgs), self.num_distractors + 1)
			target_img_idx = np.random.choice(self.num_distractors + 1)

			game_imgs = torch.stack([self.imgs[id_] for id_ in img_ids])
			target_img = game_imgs[target_img_idx]

			self.games.append({"imgs": game_imgs,
							   "labels": target_img_idx,
							   "target_img": target_img,
								}
							   )
	
	def create_labeled_data(self):
		pass

	def sample_factor(batch_size, fixed_factor, fixed_factor_value):
		factors = np.zeros([len(self._FACTORS_IN_ORDER), batch_size],
							dtype=np.int32)
		for factor, name in enumerate(self._FACTORS_IN_ORDER):
			num_choices = self._NUM_VALUES_PER_FACTOR[name]
			factors[factor] = np.random.choice(num_choices, batch_size)
		factors[fixed_factor] = fixed_factor_value
		indices = self.get_index(factors)
		
	
		return indices

	def get_index(self, factors):
		indices = 0
		base = 1
		for factor, name in reversed(list(enumerate(self._FACTORS_IN_ORDER))):
			indices += factors[factor] * base
			base *= self._NUM_VALUES_PER_FACTOR[name]
  		
		return indices

	def __len__(self):	
		return len(self.games)

	def __getitem__(self, idx):
		return self.games[idx]