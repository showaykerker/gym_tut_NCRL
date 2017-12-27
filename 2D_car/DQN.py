import random
import gym
import math
import numpy as np
import time
from collections import deque
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import rmsprop
from keras.initializers import TruncatedNormal, glorot_normal, Ones, lecun_uniform, Zeros
from keras.utils import plot_model

import matplotlib.pyplot as plt

class Color():
	W = '\033[0m'
	DR = '\033[31m'
	DG = '\033[32m'
	BG = '\033[1;32m'
	DY = '\033[33m'
	BY = '\033[1;33m'
	DB = '\033[34m'
	BB = '\033[1;34m'
	DP = '\033[35m'
	BP = '\033[1;35m'


class DQN():
	def __init__(self,
				 n_input = 5,
				 n_output = 3,
				 gamma = 0.96,
				 beta = 0.3,
				 memory_size = 2000,
				 batch_size = 32,
				 epsilon = 0.8,
				 epsilon_decay = 0.996,
				 epsilon_min = 0.02,
				 multi_thread = True,
				 show = True,
				 simple_bellman = True,
				):

		self.ep = 0

		self.gamma = gamma
		self.beta = beta
		self.batch_size = 32
		self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
		self.n_input = n_input
		self.n_output = n_output
		self.multi_thread = multi_thread
		self.memory = deque(maxlen=memory_size)
		self.loss   = deque(maxlen=100000)
		self.simple_bellman = simple_bellman

		# build model
		self.Q_eval = Sequential()
		#self.Q_eval.add(Dense(100, input_dim=n_input, activation='linear', kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
		self.Q_eval.add(Dense(100, input_dim=n_input, activation='tanh', kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
		#self.Q_eval.add(Dense(100, activation='tanh', kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
		self.Q_eval.add(Dense(100, activation='tanh', kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
		#self.Q_eval.add(Dense(100, activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
		#self.Q_eval.add(Dense(100, activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
		self.Q_eval.add(Dense(n_output, activation='linear', kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
		#self.Q_eval.add(Dense(n_output, activation='tanh', kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
		self.Q_eval.compile(optimizer = 'rmsprop', loss='mse')

		self.Q_target = clone_model(self.Q_eval)
		self.Q_target.compile(optimizer = 'rmsprop', loss='mse')

		self.Q_train = clone_model(self.Q_eval)
		self.Q_train.compile(optimizer = 'rmsprop', loss='mse')


		self.weight_of_Q_train = self.Q_train.get_weights()
		print('Neural Network Built!')

		
		


		# for plots
		if not multi_thread and show: plt.ion()
		self.fig = plt.figure()
		self.r_saver = []
		self.step_saver = []
		self.n_step_saver = deque(maxlen=100)
		self.n_r_saver = deque(maxlen=100)
		self.quartile_saver = [[],[],[]]
		self.total_time_step = 0
		self.action_taken=[0]*n_output

		self.show = show

		if multi_thread:
			print(Color.BY+'Multi-Thread Mode'+Color.W)


	def get_description(self):

		import os

		self.description = input(Color.BP+'Description: '+Color.W)
		self.test_name = input(Color.BP+'Test Name: '+Color.W)

		
		if not os.path.exists('log/' + self.test_name): os.makedirs('log/'+self.test_name)
		plot_model(self.Q_eval, to_file='log/'+self.test_name+'/'+self.description+'.png', show_shapes=True)

	def Remember(self, s, a, r, s_, done):

		self.memory.append((s, a, r, s_, done))
		self.memory.append((np.reshape(s[0][::-1], (1, self.n_input)), -a + self.n_output - 1 , r, np.reshape(s_[0][::-1], (1, self.n_input)), done))
		


	def ChooseAction(self, s):
		self.ep += 1
		ret = None
		if np.random.random() <= self.epsilon: 
			ret = random.randrange(0, self.n_output)
			'''
			if np.random.random() < 0.1:
				if np.random.random() > 0.5: ret = self.n_output-1
				else : ret = 0
			else: ret = random.randrange(1, self.n_output-1)
			'''
		else: 
			ret = np.argmax(self.Q_eval.predict(s))
		self.action_taken[ret] += 1
		return ret


	def MemoryReplay(self):
		#print('MemoryReplay!')
		x_batch, y_batch = [], []
		minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

		for s, a, r, s_, done in minibatch:
			y_target = self.Q_train.predict(s) if self.multi_thread else self.Q_eval.predict(s)
			origin = y_target[0][a]
			target = None
			if done: target = r
			else:
				if self.simple_bellman:
					#target = r + self.gamma * np.max(self.Q_target.predict(s_)[0]) 
					input('ERROR HERE!!! in DQN.py, self.MemoryReplay()')

				else:
					target = (1-self.beta) * y_target[0][a] + self.beta * (r + self.gamma * np.max(self.Q_target.predict(s_)[0]))
				

			self.loss.append(abs(target-origin))

			y_target[0][a] = target

			x_batch.append(s[0])
			y_batch.append(y_target[0]) ##

		if self.multi_thread:
			self.Q_train.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
			self.weight_of_Q_train = self.Q_train.get_weights()
		else: 
			self.Q_eval.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
		


	def ReplaceTarget(self):
		#print('ReplaceTarget!')
		weight = self.Q_train.get_weights() if self.multi_thread else self.Q_eval.get_weights()
		self.Q_target.set_weights(weight)
		
		if self.epsilon > self.epsilon_min: 
			self.epsilon *= self.epsilon_decay
			print('epsilon =', self.epsilon)
		if self.epsilon < self.epsilon_min: 
			self.epsilon = self.epsilon_min
			print('epsilon =', self.epsilon)


	def ReplaceEval(self):
		#print('ReplaceEval')
		self.Q_eval.set_weights(self.weight_of_Q_train)


	def append_data(self, ep_reward = None, ep_steps = None):
		if ep_reward is not None : 
			self.r_saver.append(ep_reward)
			self.n_r_saver.append(ep_reward)
			self.quartile_saver[0].append(np.percentile(self.n_r_saver, 25))
			self.quartile_saver[1].append(np.percentile(self.n_r_saver, 50))
			self.quartile_saver[2].append(np.percentile(self.n_r_saver, 75))
		if ep_steps is not None : 
			self.step_saver.append(ep_steps)
			self.total_time_step += ep_steps
			


	def plot(self):

		self.fig.suptitle(self.description)

		p = self.fig.add_subplot(3,1,1) 
		p.clear()
		p.set_yscale('log')
		p.set_ylabel('loss')
		p.grid(True)
		p.plot(self.loss, 'bo', ms=0.1)
		'''
		r = self.fig.add_subplot(3,1,2)
		r.clear()
		r.set_ylabel('reward per ep')
		r.grid(True)
		r.plot(self.r_saver, 'go', ms=0.1)
		'''

		s = self.fig.add_subplot(3,1,2)
		s.clear()
		s.set_ylabel('reward per ep')
		#s.set_yscale('symlog', basey=10)
		s.grid(True)
		#s.plot(self.r_saver, 'yo', ms=0.2)
		s.plot(self.quartile_saver[0], 'g-', lw=0.4)
		s.plot(self.quartile_saver[1], 'b-', lw=0.4)
		s.plot(self.quartile_saver[2], 'r-', lw=0.4)

		q = self.fig.add_subplot(3,1,3)
		q.clear()
		q.set_ylabel('action taken')
		q.grid(True)
		x = range(len(self.action_taken))
		q.bar(x, self.action_taken, 0.2, color='green')

		
		

		if not self.multi_thread and self.show: plt.pause(0.01)


	def save_model(self, ep, apd):

		import time

		while self.multi_thread and (not hasattr(self, 'test_name') or not hasattr(self, 'description')):
			time.sleep(0.5)

		time_ = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
		model_name = time_ + '_E%d' % (ep) + '_' + str(apd) + '.h5' 
		
		self.Q_eval.save('log/' + self.test_name + '/' + model_name)
		print( Color.BY + "Saving Model: " + 'log/' +  self.test_name + '/' + model_name + Color.W )

		if not self.multi_thread:
			fig_name = time_ + '_E%d' % (ep) + '_' + str(apd) + '.png' 
			self.fig.savefig('log/' + self.test_name + '/' + fig_name , dpi=self.fig.dpi)
			print( Color.BY + "Saving Figure: " + 'log/' +  self.test_name + '/' + fig_name + Color.W )

	