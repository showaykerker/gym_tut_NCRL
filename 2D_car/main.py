
import time
import numpy as np
from DQN import DQN
from car_env import CarEnv



def main(name=None, description=None):
	global env, agent, ep, TEST, PLOT

	MAX_EPISODES = 1000000
	MAX_EP_STEPS = 800
	RENDER = True


	for ep in range(MAX_EPISODES):

		s = env.reset()
		steps = 0

		for steps in range(MAX_EP_STEPS):

			if RENDER: env.render()

			# Put Your Code Here




			if done: 
				agent.append_data(ep_steps=steps)
				break




if __name__ == '__main__':

	global agent, ep, TEST
	

	env = CarEnv( map_set = 0 )


	agent = DQN(
					n_input = env.n_sensor,
					n_output = env.n_actions,
					gamma = 0.96,
					beta = 0.3,
					memory_size = 2000,
					batch_size = 32,
					epsilon = 0.8,
					epsilon_decay = 0.996,
					epsilon_min = 0.02,
					show = True,
				)

	
	agent._build_net()

	if print_only: print('Program Ended.')	
	else:
		if TEST :
			main(name='test', description=D)
		else:
			try:
				main()
			except:
				agent.save_model(ep, 'Exception')
		

