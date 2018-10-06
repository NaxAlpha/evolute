import torch as t
import numpy as np
from copy import deepcopy
from torch.nn import functional as F
from concurrent.futures import ThreadPoolExecutor as Pool

class EvoModel(t.nn.Module):
	
	def __init__(self):
		super(EvoModel, self).__init__()
		
	def remove_grad(self):
		for p in self.parameters():
			p.requires_grad_(False)
		return self
	
	def zero(self):
		for p in self.parameters():
			p.data.copy_(t.zeros(*p.shape))
		return self
	
	def noise(self, mean=0, std=1):
		for p in self.parameters():
			p.add_(t.randn(*p.shape) * std + mean)
		return self
	
	def clone(self):
		return deepcopy(self)
	
	def sample(self, idx):
		pass
	
	def evolve(self, n_pop, env, data=None):
		models = [(self.clone().noise(std=100), i) for i in range(n_pop)]
		with Pool(8) as p:
			rewards = p.map(lambda mdl: env(data, *mdl), models)
		rewards = np.array(list(rewards))
		goods = sum(rewards >= 1)
		total = sum(rewards)
		for m, r in zip(models, rewards):
			m = m[0]
			for tp, mp in zip(self.parameters(), m.parameters()):
				tp.add_(mp * 0.1 * (int(r) / int(total)))
		return goods
	
	def forward(self, *_):
		raise NotImplementedError


class FrozenModel(EvoModel):
	
	def __init__(self):
		super(FrozenModel, self).__init__()
		# self.conv1 = t.nn.Conv2d(1, 4, 3, 2, 1)
		# self.conv2 = t.nn.Conv2d(4, 16, 3, 2, 1)
		self.linr1 = t.nn.Linear(400, 4)
		self.remove_grad()
	
	def forward(self, x):
		# x.unsqueeze_(0)
		# x.unsqueeze_(0)
		# x = F.relu(self.conv1(x))
		# x = F.relu(self.conv2(x))
		x = x.view(-1)
		x = self.linr1(x)
		return x


class Trainer:
	
	def __init__(self, environ, model, render=True, max_iter=50):
		self.environ = environ
		self.model = model
		self.gen_id = 0
		self.render = render
		self.max = max_iter
		
	def evaluator(self, model, idx):
		win_name = F'Gen: {self.gen_id}, Pop: {idx}'
		env = deepcopy(self.environ)
		total_reward = 0
		count = 0
		while not env.done():
			if self.render:
				img = env.rend()
				cv2.imshow(win_name, img)
				cv2.moveWindow(win_name, 250 * (idx % 6), 250 * ((idx // 6) % 3))
				# cv2.moveWindow(win_name, 0, 0)
				cv2.waitKey(50)
			state = env.state()
			state = t.tensor(state, dtype=t.float)
			action = model.forward(state)
			action = t.argmax(action)
			total_reward = env.step(action)
			count += 1
			if self.max != 0 and count > self.max:
				total_reward = env.reward()
				break
		if self.render:
			cv2.destroyWindow(win_name)
		return total_reward
		
	def train(self, n_gen, n_pop):
		self.environ.reload()
		for _ in range(n_gen):
			# self.environ.reload()
			self.gen_id = _
			goods = self.model.evolve(n_pop, Trainer.evaluator, self)
			print(F'Gen: {_}, Goods: {goods}')
			t.save(self.model.state_dict(), 'model.pth')


if __name__ == '__main__':
	import cv2
	from evo import FrozenLake
	
	frozen = FrozenModel()
	lake = FrozenLake(20, 20, 50)
	tx = Trainer(lake, frozen, False)
	tx.train(100000, 10000)



