import gym
from stable_baselines3 import A2C
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=2000)

#run the trained policy for some
#timesteps and unknown number of episodes
obs = env.reset()
tot_rewards = []
dones = 0
rewards = 0
for i in range(200):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    rewards+=reward
    env.render()
    if done:
        tot_rewards.append(rewards)
        rewards = 0
        obs = env.reset()
        dones+=1

#episodic reward doesn't change
# over time in testing portion
plt.plot(range(dones),tot_rewards)
plt.show()

