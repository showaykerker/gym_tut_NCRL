
import time
import threading
import numpy as np

import argparse

global ep, env, agent, TEST, PLOT



def main(name=None, description=None):
	global env, agent, ep, TEST, PLOT

	MAX_EPISODES = 1000000
	MAX_EP_STEPS = 800
	RENDER = True
	SAVE_MODEL_EVERY = 500


	for ep in range(MAX_EPISODES):
		s = env.reset() # np.reshape(env.reset(), [1,env.state_dim])
		ep_step = 0
		ep_r = 0
		t = 0
		for t in range(MAX_EP_STEPS):

			if RENDER: env.render()

			a = agent.choose_action(s)
			s_, r, done = env.step(a)
			#s_ = np.reshape(s_, [1,env.state_dim])

			if a == 0 or a == agent.n_actions-1: 
				r = - 0.3
				#done = False
			elif done and t == MAX_EP_STEPS - 1: r = 0.3
			elif done and t < MAX_EP_STEPS - 1: r = - 1
			else : r = 0.3
			#else: r = 0.8 - abs(int((env.n_actions-1)/2)-a) * 0.01
			

			ep_r += r
			agent.store_transition(s, a, r, s_, done)

			s = s_
			ep_step += 1

			if done or t == MAX_EP_STEPS - 1:
				print('Ep:', ep, '| Steps: %i' % int(ep_step), ' | reward: %.1f' % ep_r, ' | time_steps: %i' % agent.total_time_step)
				break

		agent.append_data(ep_reward = ep_r, ep_steps = t) # pass data to tensorboard
		agent.learn()


		if (ep % SAVE_MODEL_EVERY == 0 and ep!=0):
			if not TEST: agent.save_model(ep, 'AutoSaved')


def get_parse():

	parser = argparse.ArgumentParser(description='showaykerker')
	parser.add_argument('-t', '--test', dest='isTest', action='store_true', help='Pass to goto test mode')
	parser.add_argument('-n', '--name', metavar='Name', dest='Name', action='store', help='Experiment name, all Data will be stored add ./logs/[Name]')
	parser.add_argument('-d', '--disc', metavar='description', dest='D', action='store', help='Add Description of the experiment.')
	parser.add_argument('-f', '--folder', metavar='FolderName', dest='FName', action='store', help='Name a specific folder where this run will create a subdirectory under it.')
	parser.add_argument('-s', '--structure', dest='print_only', action='store_true', help='Only plot model structure.')
	#parser.add_argument('-v', '--visualize', metavar='visualize', dest='Visual', action='store', choices={'tb', 'mat', 'no'}, default='tb', help='Visualization, options: tb, mat, no')
	

	args = parser.parse_args()

	try:
		if args.FName[-1] != '/': 
			args.FName += '/'
	except:
		pass

	return (args.isTest, args.Name, args.D, args.FName, args.print_only)


if __name__ == '__main__':

	global agent, ep, TEST


	TEST, Name, D, FName, print_only = get_parse()
	print('*=============================================*')
	print('  TEST        :', TEST)
	print('  Name        :', Name)
	print('  Description :', D)
	print('  Folder Name :', FName)
	print('  Print Only  :', print_only)
	print('*=============================================*')
	


	from DuelingDQN_keras import DuelingDQN
	from car_env import CarEnv


	env = CarEnv(discrete_action=True,
			 random_map=False,
			 map_set = 1 )


	agent = DuelingDQN(
			n_actions=env.n_actions,
			n_features=env.state_dim,
			learning_rate=0.001,
			gamma=0.4,
			epsilon=0.4,
			epsilon_decrease=0.001,
			epsilon_min=0.03,
			replace_target_dict={0: 50, 100:200, 600: 200, 1200: 250},
			memory_size=200000,
			batch_size=32,			
			restore_model=None,
			print_only=print_only,
			show_Q=False
	)

	
	

	agent.get_description(test = TEST, name=Name, description=D, folder=FName)
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
		

