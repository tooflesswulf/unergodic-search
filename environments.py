from typing import List, Optional, Tuple
import numpy as np
import gym
import util


class Target(util.MapObject):
    def __init__(self, name: str, x0: Tuple[int], map: np.ndarray, scale: int):
        super(Target, self).__init__(name, x0, map, scale)
        self.active = True

    def load_render(self):
        scale = self.scale
        icon = 255 * np.ones((scale, scale, 3))
        icon[util.circ_mask(scale)] = (255, 0, 0)

        from pygame import surfarray
        self.icon = surfarray.make_surface(icon)

    def collect(self):
        self.active = False

        # hide the icon
        icon = 255 * np.ones((1, 1, 3))
        from pygame import surfarray
        self.icon = surfarray.make_surface(icon)


class Agent(util.MapObject):
    def __init__(self, name: str, x0: Tuple[int], map: np.ndarray, scale: int):
        super(Agent, self).__init__(name, x0, map, scale)

        self.ksize = 25
        self.k0 = np.array([self.ksize // 2, self.ksize // 2])
        self.sensing_kernel = np.zeros((self.ksize, self.ksize))
        self.sensing_kernel[util.circ_mask(self.ksize)] = 1

    def load_render(self):
        import pygame

        img = pygame.image.load('robot-icon.png')
        img = pygame.transform.scale(img, (32, 32))

        sense_size = self.scale * self.ksize
        sense = 255 * np.ones((sense_size, sense_size, 4))
        sense[..., 3] = 0
        sense[util.circ_mask(self.scale * self.ksize)] = (0, 0, 255, 64)
        sense = util.make_surface_rgba(sense)

        shape = np.amax([img.get_size(), sense.get_size()], axis=0)
        self.icon = pygame.Surface(shape, pygame.SRCALPHA)
        self.icon.blit(sense, (shape - sense.get_size()) // 2)
        self.icon.blit(img, (shape - img.get_size()) // 2)

    def sense(self, targets: List[Target]):
        n_collect = 0
        for targ in targets:
            rel_pos = targ.get_position() - self.get_position()

            if np.any(np.abs(rel_pos) > self.ksize // 2):
                continue
            x, y = self.k0 + rel_pos
            if random.random() < self.sensing_kernel[x, y]:
                n_collect += 1
                targ.collect()

        return n_collect


class SimpleMap(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
        "px_scale": 5,
    }

    def __init__(self, num_agent: int, num_target: int, map: np.ndarray, prior: np.ndarray, render_mode: Optional[str] = None):
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

        self.targets: List[Target] = []
        self.agents: List[Agent] = []

        self.observation_space = gym.spaces.Box(low=np.zeros(prior.shape), high=np.ones(prior.shape))
        # 5 movements: N/W/S/E/none; 2 sensing options: on/off
        self.action_space = gym.spaces.Tuple([
            gym.spaces.Tuple((gym.spaces.Discrete(5), gym.spaces.Discrete(2))) for _ in range(self.num_agent)
        ])

        self.gamma = 1
        self.sensing_cost = 0

    def set_agents(self, agents: List[Agent]):
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
        for agent, (mov, sense) in zip(self.agents, action):
            dx, dy = motions[mov]
            agent.move(dx, dy)

            if sense:
                reward -= self.sensing_cost
                reward += agent.sense(self.targets)

        # Calculate sensing for all agents?
        #   0- free space
        #   1- obstacle
        #   2- another agent

        # termination condition
        terminate = not np.any([targ.active for targ in self.targets])
        return (self.map, reward, terminate, False, {})

    def reset(self):
        self.objects = []
        self.targets = []

        distr = np.ravel(self.prior)
        targets = np.random.choice(np.arange(len(distr)), replace=False, p=distr, size=self.num_target)
        targets = np.array(np.unravel_index(targets, self.prior.shape)).T

        for i, targ_loc in enumerate(targets):
            ti = Target(f't{i}', targ_loc, self.map, self.metadata['px_scale'])
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

        w, h = map.shape
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

    # print(util.circ_mask(9))
    # exit(0)

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

    env = SimpleMap(1, 50, map, prior, render_mode='human')
    aa = Agent('r0', (100, 100), env.map, env.metadata['px_scale'])
    env.set_agents([aa])
    env.reset()
    env.render()

    import random
    while True:
        dr = [(random.randint(0, 3), 1)]
        for _ in range(3):
            env.step(dr)
        env.render()
