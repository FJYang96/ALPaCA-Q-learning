from GymWrapper import render_episode
from VIAgent import MCVI
from MCTransfer import MountainCarTransfer as MCF

grav = 0.0025
gran = 100
step_limit = 200

env = MCF(gravity=grav)
a = MCVI(gran, step_limit, gravity=grav)
a.value_iteration()

render_episode(env, a)
env.close()
