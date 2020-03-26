import time

import numpy as np
import torch

from carla_random.src.dataset import preprocess_semantic
from carla_random.src.common import COLOR
from carla_random.src.carla_env import draw_traffic_lights

from team_code.base_agent import BaseAgent
from team_code.planner import RoutePlanner
from team_code.pid_controller import PIDController

from srunner.scenariomanager.carla_data_provider import CarlaActorPool
from leaderboard.autoagents.autonomous_agent import Track


def get_entry_point():
    return 'MapAgent'


class MapAgent(BaseAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self.track = Track.SENSORS
        self.initialized = False
        self.wall_start = time.time()

        # self.net = MapModel.load_from_checkpoint(path_to_conf_file)
        # self.net.cuda()
        # self.net.eval()

    def sensors(self):
        result = super().sensors()
        result.append({
            'type': 'sensor.camera.semantic_segmentation',
            'x': 0.0, 'y': 0.0, 'z': 100.0,
            'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
            'width': 512, 'height': 512, 'fov': 5 * 10.0,
            'id': 'map'
            })

        return result

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def _init(self):
        self._vehicle = CarlaActorPool.get_hero_actor()
        self._world = self._vehicle.get_world()

        self._command_planner = RoutePlanner(10.0, 200, 257)
        self._command_planner.set_route(self._global_plan, True)

        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)

        self.initialized = True

    def tick(self, input_data):
        topdown = input_data['map'][1][:, :, 2]
        topdown = draw_traffic_lights(topdown, self._vehicle)
        # topdown = topdown[:256, 128:-128]
        # topdown = preprocess_semantic(topdown)

        result = super().tick(input_data)
        result['topdown'] = topdown

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        rgb, gps, speed, topdown = self.tick(input_data)
        loc = self._vehicle.get_location()
        gps = np.array([loc.x, loc.y])

        near_node, near_command = self._waypoint_planner.run_step(gps)
        far_node, far_command = self._command_planner.run_step(gps)

        topdown = input_data['map'][1][:, :, 2]
        topdown = draw_traffic_lights(topdown, self._vehicle)
        topdown = topdown[:256, 128:-128]
        topdown = preprocess_semantic(topdown)

        transform = self._vehicle.get_transform()
        pos = np.array([transform.location.x, transform.location.y])
        theta = np.radians(90 + transform.rotation.yaw)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])
        target_world = R.T.dot(far_node - pos)
        target = 5.5 * target_world + [128, 256]
        target = [128, 256 - 128]
        heatmap = heatmap_from_point(*target)

        x = torch.cat((topdown, torch.FloatTensor(heatmap)[None]), 0)[None]
        x = x.cuda()

        out = self.net(x)
        out = (out + 1) / 2 * heatmap.shape[0]

        close = R.T.dot(near_node - pos)

        waypoints = out.cpu().squeeze().numpy() - [128, 256]
        waypoints /= 5.5

        velocity = self._vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])
        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1])

        delta = np.clip(desired_speed - speed, 0.0, 0.5)
        throttle = self._speed_controller.step(delta)
        brake = float(desired_speed < 0.1 or speed - desired_speed > 1.0)

        p = waypoints[0]
        p = close
        angle = (90 - np.degrees(np.arctan2(-p[1], p[0]))) / 90
        steer = self._turn_controller.step(angle)

        import carla

        control = carla.VehicleControl()
        # control.steer = steer
        # control.throttle = np.clip(throttle, 0.0, 0.8)
        # control.brake = brake

        from PIL import Image, ImageDraw
        import cv2

        _heatmap = Image.fromarray(np.uint8(np.stack(3 * [255 * heatmap], 2)))
        _topdown = Image.fromarray(COLOR[topdown.argmax(0).detach().cpu().numpy()])
        _rgb = Image.fromarray(rgb)
        _draw = ImageDraw.Draw(_topdown)

        for x, y in out.squeeze():
            _draw.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

        _topdown.thumbnail((256, 256))
        _heatmap.thumbnail((256, 256))
        _rgb = _rgb.resize((512, int(512 / _rgb.size[0] * _rgb.size[1])))

        _combined = Image.fromarray(np.vstack((_rgb, np.hstack((_topdown, _heatmap)))))
        _draw = ImageDraw.Draw(_combined)
        _draw.text((5, 10), 'Steer: %.3f' % steer)
        _draw.text((5, 30), 'Throttle: %.3f' % throttle)
        _draw.text((5, 50), 'Brake: %s' % brake)
        _draw.text((5, 70), 'Speed: %.3f' % speed)
        _draw.text((5, 90), 'Desired: %.3f' % desired_speed)

        cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

        return control
