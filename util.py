from typing import Tuple
import numpy as np


def circ_mask(r: int):
    out = np.zeros((r, r), dtype=bool)
    Y, X = np.ogrid[:r, :r]
    dists = np.sqrt((X+.5 - r/2)**2 + (Y+.5 - r/2)**2)
    out[dists < r / 2] = True
    return out


class MapObject:
    name: str
    icon: np.ndarray | None = None

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
