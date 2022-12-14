from typing import List, Optional, Tuple
import numpy as np
import gym
import gym_foo.util as util


class SimpleMap(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
        "px_scale": 5,
    }

    def __init__(self, num_agent: int, num_target: int,
                 map: np.ndarray, prior: np.ndarray,
                 render_mode: Optional[str] = None,
                 max_iter=None, gamma: float = 1, sensing_cost: float = 0):
        """Construction for SimpleMap environment

        Args:
            num_agent (int): Number of agents
            num_target (int): Number of targets
            map (np.array[H, W]): Map of obstacles in environment
            prior (np.array[H, W]): Location distribution for target objects
        """
        super(SimpleMap, self).__init__()
        self.render_mode = render_mode
        self.screen_width = map.shape[1] * self.metadata['px_scale']
        self.screen_height = map.shape[0] * self.metadata['px_scale']
        self.screen = None
        self.clock = None
        self.objects: List[util.MapObject] = []

        self.map = map
        self.num_target = num_target
        self.num_agent = num_agent
        self.prior = prior / np.sum(prior)

        self.targets: List[util.Target] = []
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
        self.gamma = 1
        self.sensing_cost = 0

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

        # Move all agents
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

        # termination condition
        terminate = not np.any([targ.active for targ in self.targets])

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
        self.objects = []
        self.targets = []
        self.iter = 0

        distr = np.ravel(self.prior)
        targets = np.random.choice(np.arange(len(distr)), replace=False, p=distr, size=self.num_target)
        targets = np.array(np.unravel_index(targets, self.prior.shape)).T

        for i, targ_loc in enumerate(targets):
            ti = util.Target(f't{i}', targ_loc, self.map, self.metadata['px_scale'])
            self.targets.append(ti)
            self.objects.append(ti)
        self.objects.extend(self.agents)

        for agent in self.agents:
            agent.reset()

        if self.render_mode is not None:
            self.load_render()

    def load_render(self):
        box = np.array([(0, 0), (0, 1), (1, 1), (1, 0)])
        scale = self.metadata['px_scale']
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise ImportError("pygame is not installed, run `pip install pygame`")

        if self.screen is None:
            pygame.init()
            if self.render_mode == 'human':
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.map_surface = pygame.Surface((self.screen_width, self.screen_height))
        self.map_surface.fill((255, 255, 255))

        for i, j in zip(*np.where(self.map)):
            if self.map[i, j] != 0:
                gfxdraw.filled_polygon(self.map_surface, scale * (box + (i, j)), (0, 0, 0))
        self.map_surface = pygame.transform.flip(self.map_surface, False, True)

        for obj in self.objects:
            obj.load_render()

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is not installed, run `pip install pygame`")

        w, h = self.map.shape
        scale = self.metadata['px_scale']
        self.screen.blit(self.map_surface, (0, 0))

        blits = []
        for obj in self.objects:
            surfi = obj.icon
            flip_pos = [0, h-1] + [1, -1] * obj.x
            iw, ih = obj.icon.get_size()
            offset = [iw//2 - scale//2, ih//2 - scale//2]
            blits.append((surfi, scale * flip_pos - offset))
        self.screen.blits(blits)

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.array(pygame.surfarray.pixels3d(self.screen)).transpose(1, 0, 2)


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

    env = SimpleMap(3, 50, map, prior, render_mode='human')
    a1 = util.Agent('r0', (100, 100), env.map, env.metadata['px_scale'])
    a2 = util.Agent('r1', (50, 150), env.map, env.metadata['px_scale'])
    a3 = util.Agent('r3', (150, 50), env.map, env.metadata['px_scale'])
    env.set_agents([a1, a2, a3])
    env.reset()
    # env.render()

    # traj = [(0, 1), (1, 1), (3, 1)]

    # import random

    rr = []
    import tqdm
    for _ in (pbar := tqdm.tqdm(range(1000))):
        tot_reward = 0
        env.reset()
        for _ in range(90):
            # dr = [(random.randint(0, 3), 1), (random.randint(0, 3), 1), (random.randint(0, 3), 1)]
            dr = [(0, 1), (3, 1), (1, 1)]
            # dr = [(4, 1), (4, 1), (4, 1)]
            # for _ in range(3):
            obs, reward, term, trunc, info = env.step(dr)
            tot_reward += reward
            # env.render()
        rr.append(tot_reward)
        pbar.set_description(f'mean reward: {np.mean(rr):.03f}; std: {np.std(rr) / np.sqrt(len(rr)):.03f}')
    print(f'mean reward: {np.mean(rr):.03f}; std: {np.std(rr) / np.sqrt(len(rr)):.03f}')
    print(rr)
