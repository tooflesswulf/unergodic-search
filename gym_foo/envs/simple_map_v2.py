import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
import numpy as np
import gym
import random
import gym_foo.util as util
from gym_foo.dijkstra import dijkstra

import pkg_resources
# resource_path = '/'.join(('maps', ''))

class SimpleMapAuto(gym.Env):
    def __init__(self, config):
        # num_agent: int, num_target: int,
        #          map_id=None,
        #          max_iter: int = 10_000, sensing_cost: float = 5e-2
        num_agent = config.get('num_agent', 1)
        num_target = config.get('num_target', 1)
        map_id = config.get('map_id', None)
        max_iter = config.get('max_iter', 2_000)
        sensing_cost = config.get('sensing_cost', 5e-2)

        super(SimpleMapAuto, self).__init__()

        if map_id is None:
            map_id = random.randint(0, 8)

        resource_package = __name__
        resource_path = '/'.join(('maps', f'map{map_id}.npy'))
        # path = pkg_resources.resource_string(resource_package, resource_path)
        path = pkg_resources.resource_filename(resource_package, resource_path)

        data = np.load(path, allow_pickle=True).item()
        self.map = data['map']
        self.map[self.map == 2] = 0

        self.num_target = num_target
        self.num_agent = num_agent

        self.targets = []
        self.agents = []
        self.agent_trajectories = np.zeros((*self.map.shape, num_agent))
        self.iter = 0

        obs_shape = (num_agent, *self.map.shape, 6)
        self.observation_space = gym.spaces.Box(low=np.zeros(obs_shape), high=np.ones(obs_shape))

        self.action_space = gym.spaces.Tuple([
            gym.spaces.Tuple((gym.spaces.Discrete(5), gym.spaces.Discrete(2))) for _ in range(self.num_agent)
        ])

        self.max_iter = max_iter
        self.sensing_cost = sensing_cost

        self.reset()

    def reset(self):
        '''Resets the env by re-drawing a prior distribution and randomizes agent starting locations.
        '''
        num_distrs = np.random.geometric(1 / self.num_target)
        weights = np.random.dirichlet(np.ones(num_distrs))

        uniform = np.ravel(self.map == 0).astype(int)
        uniform = uniform / np.sum(uniform)
        targ_mus = np.random.choice(np.arange(len(uniform)), replace=False, p=uniform, size=num_distrs)
        targ_mus = np.array(np.unravel_index(targ_mus, self.map.shape)).T
        targ_sigs = np.random.exponential(scale=5, size=num_distrs)

        prior = np.zeros_like(self.map, dtype=float)
        for i, w in enumerate(weights):
            mu = targ_mus[i]
            sig = targ_sigs[i]
            d = dijkstra(self.map, mu, max_dist=3*sig) / sig
            p = np.exp(-d*d/2)
            p[d < 0] = 0
            p = p / np.sum(p)
            prior += w * p

        self.prior = prior
        self.target_distr = np.copy(self.prior)
        distr = np.ravel(self.prior)
        targets = np.random.choice(np.arange(len(distr)), replace=True, p=distr, size=self.num_target)
        targets = np.array(np.unravel_index(targets, self.prior.shape)).T
        for i, targ_loc in enumerate(targets):
            ti = util.Target(f't{i}', targ_loc, self.map, 0)
            self.targets.append(ti)

        agent_starts = np.random.choice(np.arange(len(uniform)), replace=False, p=uniform, size=self.num_agent)
        agent_starts = np.array(np.unravel_index(agent_starts, self.map.shape)).T

        self.agents = []
        for pos in agent_starts:
            a = util.Agent('agent', pos, self.map)
            a.ksize = 3
            a.sensing_kernel = np.array([[.2, .2, .2], [.2, .5, .2], [.2, .2, .2]])
            self.agents.append(a)

        self.agent_trajectories = np.zeros((*self.map.shape, self.num_agent))
        self.iter = 0
        return self._obs()

    def step(self, action: List[Tuple[int]]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Each agent takes 1 step and potentially senses the environment. 

        Returns:
            obs (Tuple[spaces.Box, List[Int[2]]]): Game observation. 
            reward (float): reward
            terminated (bool): game termination (terminal state of MDP)
            truncated (bool): early exit (e.g. time limit)
            info (dict): ???
        """

        motions = [(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)]
        reward = 0

        for i, (agent, (mov, sense)) in enumerate(zip(self.agents, action)):
            dx, dy = motions[mov]
            agent.move(dx, dy)

            if sense:
                reward -= self.sensing_cost
                reward += agent.sense(self.targets)

                xi, yi = agent.get_position()
                self.agent_trajectories[xi, yi, i] += 1

        self.iter += 1
        trunc = (self.max_iter is not None) and (self.iter > self.max_iter)
        found_all = not np.any([targ.active for targ in self.targets])

        terminate = trunc and found_all
        return self._obs(), reward, terminate, {}


    def _obs(self):
        agent_locs_map = np.zeros(self.map.shape)
        for i in range(self.num_agent):
            xi, yi = self.agents[i].get_position()
            agent_locs_map[xi, yi] = 1

        # observation channels:
        #   0: prior distr
        #   1: map & walls
        #   2: current position
        #   3: neighbors
        #   4: previous trajectory
        #   5: neighbor previous trajectory

        observations = []
        for i in range(self.num_agent):
            xi, yi = self.agents[i].get_position()
            observation = np.zeros((*self.map.shape, 6))
            observation[..., 0] = self.prior
            observation[..., 1] = self.map
            observation[xi, yi, 2] = 1
            observation[..., 3] = agent_locs_map - observation[..., 2]
            observation[..., 4] = self.agent_trajectories[..., i]
            if np.sum(observation[..., 4]) > 0:
                observation[..., 4] = observation[..., 4] / np.sum(observation[..., 4])
            observation[..., 5] = np.sum(self.agent_trajectories, axis=-1) - self.agent_trajectories[..., i]
            if np.sum(observation[..., 5]) > 0:
                observation[..., 5] = observation[..., 5] / np.sum(observation[..., 5])

            observations.append(observation)

        return np.array(observations)
