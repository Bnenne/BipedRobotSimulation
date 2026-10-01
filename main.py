from world import WorldEnv
from tqdm import tqdm

XML = None
with open('./world.xml', 'rb') as f:
    XML = f.read()

epochs = 100
current_epoch = 0

max_steps = 1000000

env = WorldEnv(XML, max_steps)

with tqdm(total=epochs, desc="Training") as pbar:
    with env.get_viewer() as viewer:
        while viewer.is_running():

            obs = env.reset()
            done = False

            while not done:
                env.step([0.001, 0.001])

            pbar.update(1)
            current_epoch += 1