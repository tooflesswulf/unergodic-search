from typing import List, Optional, Tuple
import numpy as np
import gym
import gym_foo.util as util


class ExpectationMap(gym.Env):
    def __init__(self, num_agent: int, num_target: int,
                 map: np.ndarray, prior: np.ndarray,
                 max_iter: int = None, gamma: float = 1, sensing_cost: float = 0):
        super(ExpectationMap, self).__init__()

        self.map = map
        self.num_target = num_target
        self.num_agent = num_agent
        self.prior = prior / np.sum(prior)
        self.target_distr = np.copy(prior)

        self.agents: List[util.Agent] = []
        self.agent_trajectories = np.zeros((*map.shape, num_agent))
        self.iter = 0

        obs_shape = (num_agent, *prior.shape, 6)
        self.observation_space = gym.spaces.Box(low=np.zeros(obs_shape), high=np.ones(obs_shape))
        # 5 movements: N/W/S/E/none; 2 sensing options: on/off
        self.action_space = gym.spaces.Tuple([
            gym.spaces.Tuple((gym.spaces.Discrete(5), gym.spaces.Discrete(2))) for _ in range(self.num_agent)
        ])

        self.max_iter = max_iter
        self.gamma = gamma
        self.sensing_cost = sensing_cost

    def set_agents(self, agents: List[util.Agent]):
        self.agents = agents

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
                kern = agent.sensing_kernel
                h, w = kern.shape
                xi, yi = agent.get_position()

                kern_padded = np.pad(np.zeros_like(self.target_distr), agent.k0)
                kern_padded[xi:xi+h, yi:yi+w] = kern
                k1, k2 = agent.k0
                kern_padded = kern_padded[k1:-k1, k2:-k2]

                score = np.sum(kern_padded * self.target_distr)
                reward += score * self.num_target
                self.target_distr = self.target_distr * (1 - kern_padded)

                self.agent_trajectories[xi, yi, i] += 1

        self.iter += 1
        trunc = (self.max_iter is not None) and (self.iter > self.max_iter)

        terminate = False

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
            observation[..., 4] = observation[..., 4] / np.sum(observation[..., 4])
            observation[..., 5] = np.sum(self.agent_trajectories, axis=-1) - self.agent_trajectories[..., i]
            observation[..., 5] = observation[..., 5] / np.sum(observation[..., 5])

            observations.append(observation)

        return (np.array(observations), reward, terminate, trunc, {})

    def reset(self):
        self.target_distr = np.copy(self.prior)
        for agent in self.agents:
            agent.reset()


if __name__ == '__main__':
    dims = (200, 200)

    map = np.zeros(dims)
    map[20:30, 100:] = 1
    map[90:100, :100] = 1
    map[:10, :10] = 1

    roi = [(1, 200, 200, 30), (3, 50, 150, 80)]
    prior = np.zeros(dims)
    xx, yy = np.mgrid[:dims[0], :dims[1]]
    for w, cx, cy, sig in roi:
        prior += np.exp(-((xx - cx)**2 + (yy-cy)**2) / 2/sig/sig)
    prior[map != 0] = 0

    env = ExpectationMap(3, 50, map, prior)
    a1 = util.Agent('r0', (100, 100), env.map)
    a2 = util.Agent('r1', (50, 150), env.map)
    a3 = util.Agent('r3', (150, 50), env.map)
    env.set_agents([a1, a2, a3])
    env.reset()

    tot_reward = 0
    for _ in range(90):
        dr = [(0, 1), (3, 1), (1, 1)]
        # dr = [(4, 1), (4, 1), (4, 1)]
        obs, reward, term, trunc, info = env.step(dr)
        tot_reward += reward

    print(f'total expected reward: {tot_reward}')
