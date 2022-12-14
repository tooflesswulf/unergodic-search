from typing import Tuple, List
import numpy as np
import random


def circ_mask(r: int):
    out = np.zeros((r, r), dtype=bool)
    Y, X = np.ogrid[:r, :r]
    dists = np.sqrt((X+.5 - r/2)**2 + (Y+.5 - r/2)**2)
    out[dists < r / 2] = True
    return out


def make_surface_rgba(array):
    """Returns a surface made from a [w, h, 4] numpy array with per-pixel alpha
    """
    shape = array.shape
    if len(shape) != 3 and shape[2] != 4:
        raise ValueError("Array not RGBA")

    import pygame
    import pygame.pixelcopy
    # Create a surface the same width and height as array and with
    # per-pixel alpha.
    surface = pygame.Surface(shape[0:2], pygame.SRCALPHA, 32)
    array = array.astype(int)

    # Copy the rgb part of array to the new surface.
    pygame.pixelcopy.array_to_surface(surface, array[:, :, :3])

    # Copy the alpha part of array to the surface using a pixels-alpha
    # view of the surface.
    surface_alpha = np.array(surface.get_view('A'), copy=False)
    surface_alpha[:, :] = array[:, :, 3]

    return surface


class MapObject:
    name: str
    icon: np.ndarray = None

    def __init__(self, name: str, x0: Tuple[int], map: np.ndarray, scale: int):
        self.name = name

        self.x0 = np.array(x0)
        self.x = np.array(x0)
        self.map = map

        self.scale = scale

    def reset(self):
        self.x = np.copy(self.x0)

    def move(self, dx: int, dy: int):
        """Moves the Object by dx, dy. Collisions are treated as invalid moves and leave the object in-place."""
        x2, y2 = self.x + [dx, dy]
        if x2 < 0 or y2 < 0 or x2 >= self.map.shape[0] or y2 >= self.map.shape[1]:
            return
        if self.map[x2, y2] != 0:
            return

        self.x = np.array([x2, y2])

    def get_position(self):
        return self.x

    def set_position(self, x):
        self.x = x

    def load_render(self):
        scale = self.scale
        icon = 255 * np.ones((scale, scale, 3))
        icon[circ_mask(scale)] = 0

        from pygame import surfarray
        self.icon = surfarray.make_surface(icon)


class Target(MapObject):
    def __init__(self, name: str, x0: Tuple[int], map: np.ndarray, scale: int):
        super(Target, self).__init__(name, x0, map, scale)
        self.active = True

    def load_render(self):
        scale = self.scale
        icon = 255 * np.ones((scale, scale, 3))
        icon[circ_mask(scale)] = (255, 0, 0)

        from pygame import surfarray
        self.icon = surfarray.make_surface(icon)

    def collect(self):
        self.active = False

        # hide the icon
        icon = 255 * np.ones((self.scale, self.scale, 3))
        icon[circ_mask(self.scale)] = (0, 255, 0)
        from pygame import surfarray
        self.icon = surfarray.make_surface(icon)


class Agent(MapObject):
    def __init__(self, name: str, x0: Tuple[int], map: np.ndarray, scale: int=0):
        super(Agent, self).__init__(name, x0, map, scale)

        self.ksize = 25
        self.k0 = np.array([self.ksize // 2, self.ksize // 2])
        self.sensing_kernel = np.zeros((self.ksize, self.ksize))
        self.sensing_kernel[circ_mask(self.ksize)] = .01

    def load_render(self):
        import pygame

        img = pygame.image.load('robot-icon.png')
        img = pygame.transform.scale(img, (32, 32))

        sense_size = self.scale * self.ksize
        sense = 255 * np.ones((sense_size, sense_size, 4))
        sense[..., 3] = 0
        sense[circ_mask(self.scale * self.ksize)] = (0, 0, 255, 64)
        sense = make_surface_rgba(sense)

        shape = np.amax([img.get_size(), sense.get_size()], axis=0)
        self.icon = pygame.Surface(shape, pygame.SRCALPHA)
        self.icon.blit(sense, (shape - sense.get_size()) // 2)
        self.icon.blit(img, (shape - img.get_size()) // 2)

    def sense(self, targets: List[Target]):
        n_collect = 0
        for targ in targets:
            if not targ.active:
                continue
            rel_pos = targ.get_position() - self.get_position()

            if np.any(np.abs(rel_pos) > self.ksize // 2):
                continue
            x, y = self.k0 + rel_pos
            if random.random() < self.sensing_kernel[x, y]:
                n_collect += 1
                targ.collect()

        return n_collect
