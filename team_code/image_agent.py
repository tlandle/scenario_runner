import time

import numpy as np
import cv2
import torch
import torchvision
import carla

from PIL import Image, ImageDraw

from team_code.carla_project.src.dataset import make_heatmap
from team_code.carla_project.src.models import SegmentationModel
from team_code.carla_project.src.image_model import ImageModel
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


        self.net = ImageModel.load_from_checkpoint(path_to_conf_file)
        self.net.eval()
        self.net.cuda()
        self.net.freeze()

        self.track = Track.SENSORS
        self.initialized = False
        self.wall_start = time.time()
        self.converter = Converter()

    def _init(self):
        self._command_planner = RoutePlanner(10.0, 200, 512)
        self._command_planner.set_route(self._global_plan, True)

        self._turn_controller = PIDController(K_P=0.7, K_I=0.75, K_D=0.4, n=40)
        self._speed_controller = PIDController(K_P=0.5, K_I=0.75, K_D=0.4, n=40)

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

        img = torch.cat(
                tuple(torchvision.transforms.functional.to_tensor(tick_data[x])
                    for x in ['rgb', 'rgb_left', 'rgb_right']))

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

        out, (target_cam, _) = self.net.forward(img[None].cuda(), torch.from_numpy(command_target)[None].cuda())
        target_cam = target_cam.squeeze()
        control = self.net.controller(out).cpu().squeeze()
        out = out.cpu().squeeze()

        # out = self.net(x.cuda()).cpu()
        # out[..., 0] = (out[..., 0] + 1) / 2 * rgb.shape[2]
        # out[..., 1] = (out[..., 1] + 1) / 2 * rgb.shape[1]
        # out = out.squeeze()
        # points_world = self.converter.cam_to_world(out).numpy()

        # aim = (points_world[1] + points_world[0]) / 2.0
        # angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        # steer = self._turn_controller.step(angle)
        # steer = np.clip(steer, -1.0, 1.0)
        steer = control[0].item()

        speed = tick_data['speed']
        # desired_speed = np.linalg.norm(np.mean(points_world[:-1] - points_world[1:], 0)) * 2.0
        # desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0
        # desired_speed *= (1 - abs(angle)) ** 2
        desired_speed = control[1].item()

        delta = np.clip(desired_speed - speed, 0.0, 0.4)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.7)
        brake = desired_speed < 0.2 or speed - desired_speed > 0.5

        if brake:
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        _rgb = Image.fromarray(tick_data['rgb'])
        _draw_rgb = ImageDraw.Draw(_rgb)
        _draw_rgb.ellipse((target_cam[0]-3,target_cam[1]-3,target_cam[0]+3,target_cam[1]+3), (255, 255, 255))

        for x, y in out:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 144

            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

        _combined = Image.fromarray(np.hstack([tick_data['rgb_left'], _rgb, tick_data['rgb_right']]))
        _draw = ImageDraw.Draw(_combined)
        _draw.text((5, 10), 'Steer: %.3f' % steer)
        _draw.text((5, 30), 'Throttle: %.3f' % throttle)
        _draw.text((5, 50), 'Brake: %s' % brake)
        _draw.text((5, 70), 'Speed: %.3f' % speed)
        _draw.text((5, 90), 'Desired: %.3f' % desired_speed)

        cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

        _combined.save('/tmp/video/%04d.png' % self.step)

        return control
