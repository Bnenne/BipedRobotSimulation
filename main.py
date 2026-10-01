from world import WorldEnv

XML = None
with open('./world.xml', 'rb') as f:
  XML = f.read()

env = WorldEnv(XML)

with env.get_viewer() as viewer:
    while viewer.is_running():
        env.step([0.001, 0.001])