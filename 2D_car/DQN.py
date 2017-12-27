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
				 show = True,
				):

		#####################################################################################
		# DON'T EDIT THESE BLOCK IF YOU ARE NOT SURE WHAT IT IS DOING
		self.ep = 0
		self.gamma = gamma
		self.beta = beta
		self.batch_size = 32
		self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
		self.n_input = n_input
		self.n_output = n_output
		self.loss   = deque(maxlen=100000)
		self.plot_on = {'reward': False, 'S=steps:':False}
		# DON'T EDIT THESE BLOCK IF YOU ARE NOT SURE WHAT IT IS DOING
		#####################################################################################



		self.memory = deque(maxlen=memory_size)  # This is where to store your memory



		# Build Your Model "self.Q_eval" HERE ! 
		self.Q_eval = Sequential()
		





		


		# make sure to compile your model !


		# Copy Q_eval to target net Q_target and compile.
		self.Q_target = clone_model(self.Q_eval)
		self.Q_target.compile(optimizer = 'Adadelta', loss='mse')


		self.weight_of_Q_train = self.Q_train.get_weights()
		print('Neural Network Built!')

		
		

		#####################################################################################
		# DON'T EDIT THESE BLOCK IF YOU ARE NOT SURE WHAT IT IS DOING
		# for plots
		if show: plt.ion()
		self.fig = plt.figure()
		self.r_saver = []
		self.step_saver = []
		self.n_step_saver = deque(maxlen=100)
		self.n_r_saver = deque(maxlen=100)
		self.quartile_saver = [[],[],[]]
		self.total_time_step = 0
		self.action_taken=[0]*n_output
		self.show = show
		# DON'T EDIT THESE BLOCK IF YOU ARE NOT SURE WHAT IT IS DOING
		#####################################################################################



	def Remember(self, s, a, r, s_, done):
		# Store it in your memory
		pass
				
		


	def ChooseAction(self, s):
		self.ep += 1
		# Choose action and return.
		



	def MemoryReplay(self):		

		x_batch, y_batch = [], []
		minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

		for s, a, r, s_, done in minibatch:
			# I've sample the minibatch from memory, the rest is your work !
			# You can do "DQN" rather than "Double DQN" first
			# You can use "self.gamma" you passed in at main.py in Bellman Equation 
			pass


		
		self.Q_eval.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
		





	def ReplaceTarget(self): # You Don't Need To Edit Here

		# Call this whenever you want to replace your Target Net (The freezed one) by Eval Net
		# If you are trying "DQN" rather than "Double DQN", just ignore this

		weight = self.Q_train.get_weights() if self.multi_thread else self.Q_eval.get_weights()
		self.Q_target.set_weights(weight)
		
		if self.epsilon > self.epsilon_min: 
			self.epsilon *= self.epsilon_decay
			print('epsilon =', self.epsilon)
		if self.epsilon < self.epsilon_min: 
			self.epsilon = self.epsilon_min
			print('epsilon =', self.epsilon)



	def append_data(self, ep_reward = None, ep_steps = None): # You Don't Need To Edit Here
		# Record Data for Result Plotting
		if ep_reward is not None : 
			self.plot_on['reward'] = True
			self.r_saver.append(ep_reward)
			self.n_r_saver.append(ep_reward)
			self.quartile_saver[0].append(np.percentile(self.n_r_saver, 25))
			self.quartile_saver[1].append(np.percentile(self.n_r_saver, 50))
			self.quartile_saver[2].append(np.percentile(self.n_r_saver, 75))
		if ep_steps is not None : 
			self.plot_on['steps'] = True
			self.step_saver.append(ep_steps)
			self.total_time_step += ep_steps
			


	def plot(self): # You Don't Need To Edit Here
		# Plot Result using matplotlib

		if self.show: 

			if self.plot_on['reward'] and self.plot_on['steps']:

				self.fig.suptitle('')

				r = self.fig.add_subplot(2,1,1)
				r.clear()
				r.set_ylabel('reward per ep')
				#r.set_yscale('symlog', basey=10)
				r.grid(True)
				#r.plot(self.r_saver, 'yo', ms=0.2)
				r.plot(self.quartile_saver[0], 'g-', lw=0.4)
				r.plot(self.quartile_saver[1], 'b-', lw=0.4)
				r.plot(self.quartile_saver[2], 'r-', lw=0.4)

				s = self.fig.add_subplot(2,1,2)
				s.clear()
				s.set_ylabel('step per ep')
				#s.set_yscale('log')
				s.grid(True)
				s.plot(self.step_saver, 'b-', lw=0.4)

				plt.pause(0.01)

			elif self.plot_on['reward'] is False and self.plot_on['steps'] is True:
				
				self.fig.suptitle('')

				s = self.fig.add_subplot(1,1,1)
				s.clear()
				s.set_ylabel('step per ep')
				#s.set_yscale('log')
				s.grid(True)
				s.plot(self.step_saver, 'b-', lw=0.4)


			elif self.plot_on['reward'] is False and self.plot_on['steps'] is True:

				self.fig.suptitle('')

				r = self.fig.add_subplot(1,1,1)
				r.clear()
				r.set_ylabel('reward per ep')
				#r.set_yscale('symlog', basey=10)
				r.grid(True)
				#r.plot(self.r_saver, 'yo', ms=0.2)
				r.plot(self.quartile_saver[0], 'g-', lw=0.4)
				r.plot(self.quartile_saver[1], 'b-', lw=0.4)
				r.plot(self.quartile_saver[2], 'r-', lw=0.4)

			elif self.plot_on['reward'] is False and self.plot_on['steps'] is False:
				pass

				

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