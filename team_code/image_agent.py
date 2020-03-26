import time

import numpy as np
import cv2
import torch
import torchvision
import carla

from PIL import Image, ImageDraw

from team_code.carla_project.src.dataset import make_heatmap
from team_code.carla_project.src.models import SegmentationModel
from team_code.carla_project.src.converter import Converter

from team_code.base_agent import BaseAgent
from team_code.planner import RoutePlanner
from team_code.pid_controller import PIDController

from leaderboard.autoagents.autonomous_agent import Track


def get_entry_point():
    return 'ImageAgent'


class ImageAgent(BaseAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self.net = SegmentationModel(4, 4)
        self.net.load_state_dict(torch.load(path_to_conf_file))
        self.net.eval()
        self.net.cuda()

        self.track = Track.SENSORS
        self.initialized = False
        self.wall_start = time.time()
        self.converter = Converter()

    def _init(self):
        self._command_planner = RoutePlanner(10.0, 200, 512)
        self._command_planner.set_route(self._global_plan, True)

        self._turn_controller = PIDController(K_P=0.8, K_I=0.75, K_D=0.4, n=40)
        self._speed_controller = PIDController(K_P=0.5, K_I=1.0, K_D=1.0, n=40)

        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale

        return gps

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)
        gps = self._get_position(tick_data)
        far_node, far_command = self._command_planner.run_step(gps)

        rgb = tick_data['rgb']
        rgb = torchvision.transforms.functional.to_tensor(rgb)

        theta = tick_data['compass']
        if np.isnan(theta):
            theta = 0.0
        theta = theta + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])
        command_target = R.T.dot(far_node - gps)
        command_target *= 5.5
        command_target += [128, 256]
        command_target = np.clip(command_target, 0, 256)

        command_img = self.converter.map_to_cam(torch.FloatTensor(command_target))
        heatmap_img = make_heatmap((144, 256), command_img)
        heatmap_img = torch.FloatTensor(heatmap_img).unsqueeze(0)

        x = torch.cat((rgb, heatmap_img), 0)[None]
        out = self.net(x.cuda()).cpu()
        out[..., 0] = (out[..., 0] + 1) / 2 * rgb.shape[2]
        out[..., 1] = (out[..., 1] + 1) / 2 * rgb.shape[1]
        out = out.squeeze()
        points_world = self.converter.cam_to_world(out).numpy()

        aim = (points_world[1] + points_world[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        speed = tick_data['speed']
        # desired_speed = np.linalg.norm(np.mean(points_world[:-1] - points_world[1:], 0)) * 2.0
        desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0
        desired_speed *= (1 - abs(angle)) ** 2

        delta = np.clip(desired_speed - speed, 0.0, 0.5)
        throttle = self._speed_controller.step(delta)
        brake = desired_speed < 0.25 or speed - desired_speed > 0.25

        if brake:
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = np.clip(throttle, 0.0, 0.8)
        control.brake = float(brake)

        _heatmap = Image.fromarray(np.uint8(np.stack(3 * [255 * heatmap_img.squeeze()], 2)))
        _rgb = Image.fromarray(np.uint8(rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255))
        _draw_rgb = ImageDraw.Draw(_rgb)

        for x, y in out:
            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

        _combined = Image.fromarray(np.hstack((_rgb, _heatmap)))
        _draw = ImageDraw.Draw(_combined)
        _draw.text((5, 10), 'Steer: %.3f' % steer)
        _draw.text((5, 30), 'Throttle: %.3f' % throttle)
        _draw.text((5, 50), 'Brake: %s' % brake)
        _draw.text((5, 70), 'Speed: %.3f' % speed)
        _draw.text((5, 90), 'Desired: %.3f' % desired_speed)
        _draw.text((5, 110), 'Angle: %.3f' % angle)

        cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

        return control
