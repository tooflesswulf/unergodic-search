from dataclasses import dataclass, field
from typing import Any
import numpy as np
import queue


@dataclass(order=True)
class PQItem:
    dist: int
    item: Any = field(compare=False)


dxy = [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,1), (1,-1), (-1,-1)]
def dijkstra(map, start, max_dist=10):
    '''map is 0/1 for free/obstacle. Computes distances w/ 4-connectivity.
    '''

    out = np.zeros_like(map) - 1
    x0, y0 = start

    pq = queue.PriorityQueue()
    pq.put(PQItem(0, (x0, y0)))
    while not pq.empty():
        pqi = pq.get()
        d = pqi.dist
        x, y = pqi.item
        if out[x, y] >= 0 and out[x, y] <= d:
            continue
        out[x, y] = d

        if d >= max_dist:
            continue

        for dx, dy in dxy:
            nx, ny = x+dx, y+dy
            if map[nx, ny] != 0: continue
            if out[nx, ny] < 0:
                pq.put(PQItem(d + 1, (nx, ny)))

    return out


