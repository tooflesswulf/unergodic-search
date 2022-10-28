from environments import SimpleMap
from util import Agent
import numpy as np


if __name__ == '__main__':
    data = np.load('demo-1.npz')
    map = data['map']
    distr = data['distr']
    dims = map.shape

    env = SimpleMap(1, 50, map, distr, render_mode='human')

    aa = Agent('r0', (150, 150), env.map, env.metadata['px_scale'])
    env.set_agents([aa])

    env.reset()
    env.render()

    import random
    while True:
        dr = [(random.randint(0, 3), 1)]
        for _ in range(3):
            env.step(dr)
        env.render()
