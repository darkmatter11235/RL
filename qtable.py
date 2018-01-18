import gym
import numpy as np

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])

lr = 0.8
y = 0.95
num_episodes = 2000

#reward list
rList = []
for i in range(num_episodes):
    #Reset environment and get the first episode
    s = env.reset()
    rAll = 0
    d = False
    j = 0

    #The Q-table learning loop
    while j < 99:
        j += 1
        #Choose an action with maximum reward (which some stochastic noise)
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get the new state
        s1,r,d,_ = env.step(a)
        #Update Q-Table 
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break

    rList.append(rAll)

print "Score over time: " + str(sum(rList)/num_episodes)

print "Final Q-Table Values"
print Q
