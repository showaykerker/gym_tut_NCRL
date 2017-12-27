import gym

env = gym.make('CartPole-v0')

env.reset()                                           # The method "reset()" returns the value of initial state

for _ in range(1000):
	#env.render()                                 # Shows the window to display the environment 'env'
	action = env.action_space.sample()            # Randomly select an action
	env.step(action)                              # The method "step(action)" applies action you pass in, and returns (s_, r, done, info)


############################################
#                                          #
# Exercise:                                #
#                                          #
# 1. Try to uncomment env.render().        #
#                                          #
# 2. Try to print out the initial state.   #
#                                          #
# 3. Try to apply different actions.       #
#                                          #
############################################
