import gym

#Setting MountainCar-v0 as the environment
env = gym.make("MountainCar-v0", render_mode="rgb_array")

#Sets an initial state
env.reset()

# Rendering our instance 300 timesco
for _ in range(300):
    #Takes a random action from its action space 
    # aka the number of unique actions an agent can perform
    env.step(env.action_space.sample())
    #renders the environment
    env.render()

env.close()