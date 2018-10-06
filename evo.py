# indented with tab
import cv2
import math
import numpy as np
from random import randrange as rnd


class Environ:
	
	def __init__(self):
		self._state = None
	
	def state(self):
		return self._state
	
	def done(self):
		pass
	
	def reset(self):
		pass
	
	def step(self, action):
		pass
	
	def reward(self):
		pass
	
	def rend(self):
		pass


def test_agent(env, agent):
	total_reward = 0
	while not env.done():
		state = env.state()
		action = agent(state)
		reward = env.step(action)
		total_reward += reward
	return total_reward


def evolve(agents, reward):
	return None, []


def evolute(agent_sample, environ_sample, npop, ngens):
	best = None
	agents = [agent_sample.copy() for __ in range(npop)]
	for _ in range(ngens):
		rewards = []
		for agent in agents:
			environ_sample.reset()
			total_reward = test_agent(environ_sample, agent)
			rewards.append(total_reward)
		best, agents = evolve(agents, rewards)
	return best


class FrozenLake(Environ):

	def __init__(self, w, h, x):
		self.w = w
		self.h = h
		self.x = x
		self.d = 0
		self.pos = (0, 0)
		self.init = (0, 0)
		self.reset()
		
	def rand(self, val):
		x, y = rnd(self.w), rnd(self.h)
		while self._state[y, x] != 0:
			x, y = rnd(self.w), rnd(self.h)
		self._state[y, x] = val
		return x, y

	def reload(self):
		h, w = self.h, self.w
		self._state = np.zeros((h, w), dtype=np.uint8)
		for _ in range(self.x):
			self.rand(1)  # Bad pole
		self.init = self.rand(2)  # Player
		self.pos = self.init
		self.rand(3)  # Goal
		self.d = 0
		self.n = 0

	def reset(self):
		self.pos = self.init
		self.n = 0
	
	def reward(self):
		return 0 
		x0, y0 = self.init
		x1, y1 = self.pos
		x2, y2 = np.argwhere(self._state == 3)[0]
		ds = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)  # Distance from start
		dt = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # Distance Travelled
		df = math.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2)  # Distance from final
		rm = math.sqrt(self.w ** 2 + self.h ** 2)        # Max possible "ds"
		return ds / (self.n * rm * (dt + 1) * (df + 1))  # Normalized Current Reward

	def move(self, delta):
		self.n += 1
		dx, dy = delta
		x, y = self.pos
		if  (dx < 0 and x == 0) or \
			(dy < 0 and y == 0) or \
			(dx > 0 and x == self.w - 1) or \
			(dy > 0 and y == self.h - 1):
			return 0
		state = self._state[y+dy, x+dx]
		if state == 3:
			self.d = 1
			return 1 + self.reward()
		if state == 1:
			self.d = 1
			return -(1 + self.reward())
		self._state[y, x] = 0
		self._state[y+dy, x+dx] = 2
		self.pos = x+dx, y+dy
		return 0

	def step(self, action):
		actions = [
			(-1, 0),  # Left
			(+1, 0),  # Right
			(0, -1),  # Up
			(0, +1),  # Down
		]
		return self.move(actions[action])
	
	def done(self):
		return self.d

	def rend(self):
		clrs = [
			(255, 255, 255),  # Flat
			(0, 0, 255),  # Danger
			(255, 0, 0),  # Player
			(0, 255, 0),  # Success
		]
		sw, sh = 10, 10
		img = np.zeros((self.h * sh, self.w * sw, 3), dtype=np.uint8)
		for y in range(self.h):
			for x in range(self.w):
				img[(sh*y):(sh*(y+1)), (sw*x):(sw*(x+1))] = clrs[self._state[y, x]]
		return img




			


		
		


