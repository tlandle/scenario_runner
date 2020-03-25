from collections import deque

import numpy as np


class Plotter(object):
    def __init__(self, size):
        self.size = size
        self.clear()
        self.title = str(self.size)

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, x, y, color=(255, 255, 255), r=2):
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x-r, y-r, x+r, y+r), color)

    def show(self):
        import cv2

        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance

        self.debug = Plotter(debug_size)

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        self.mean = np.array([49.0, 8.0])
        self.scale = np.array([111324.60662786, 73032.1570362])

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.route.append((pos, cmd))

    def run_step(self, gps):
        self.debug.clear()

        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)

            u = (self.route[i][0] - gps) * 5.5

            if distance > self.min_distance:
                self.debug.dot(u[0], u[1], (255, 255 * int(self.route[i][1].value == 4), 255))
            else:
                self.debug.dot(u[0], u[1], (0, 255 * int(self.route[i][1].value == 4), 255))

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        u = (self.route[0][0] - gps) * 5.5
        self.debug.dot(u[0], u[1], (0, 255, 0))

        u = (self.route[1][0] - gps) * 5.5
        self.debug.dot(u[0], u[1], (255, 0, 0))

        u = (gps - gps) * 5.5
        self.debug.dot(u[0], u[1], (0, 0, 255), 1)
        # self.debug.show()

        return self.route[1]
