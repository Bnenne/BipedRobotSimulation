from stable_baselines3 import SAC
from world import WorldEnv

with open('./world.xml', 'rb') as f:
    XML = f.read()

env = WorldEnv(XML, max_steps=2000)

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000, progress_bar=True)
model.save("sac_world")