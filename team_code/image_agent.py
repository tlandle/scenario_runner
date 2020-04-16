import numpy as np
import cv2
import torch
import torchvision
import carla

from PIL import Image, ImageDraw

from carla_project.src.image_model import ImageModel
from carla_project.src.converter import Converter

from team_code.base_agent import BaseAgent
from team_code.pid_controller import PIDController


DEBUG = True


def get_entry_point():
    return 'ImageAgent'


def debug_display(tick_data, target_cam, out, steer, throttle, brake, desired_speed, step):
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
    _draw.text((5, 70), 'Speed: %.3f' % tick_data['speed'])
    _draw.text((5, 90), 'Desired: %.3f' % desired_speed)

    cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)

    # _combined.save('/tmp/video/%04d.png' % step)


class ImageAgent(BaseAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self.converter = Converter()
        self.net = ImageModel.load_from_checkpoint(path_to_conf_file)
        self.net.cuda()
        self.net.eval()

    def _init(self):
        super()._init()

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)
        gps = self._get_position(tick_data)
        far_node, _ = self._command_planner.run_step(gps)

        img = torch.cat(tuple(
            torchvision.transforms.functional.to_tensor(tick_data[x])
            for x in ['rgb', 'rgb_left', 'rgb_right']))[None].cuda()

        theta = tick_data['compass']
        theta = 0.0 if np.isnan(theta) else theta
        theta = theta + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        command_target = R.T.dot(far_node - gps)
        command_target *= 5.5
        command_target += [128, 256]
        command_target = np.clip(command_target, 0, 256)
        command_target = torch.from_numpy(command_target)[None].cuda()

        out, (target_cam, _) = self.net.forward(img, command_target)
        control = self.net.controller(out).cpu().squeeze()
        out = out.cpu().squeeze()
        target_cam = target_cam.squeeze()

        steer = control[0].item()
        desired_speed = control[1].item()
        speed = tick_data['speed']

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        if brake:
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        if DEBUG:
            debug_display(
                    tick_data, target_cam, out,
                    steer, throttle, brake, desired_speed,
                    self.step)

        return control

# out = self.net(x.cuda()).cpu()
# out[..., 0] = (out[..., 0] + 1) / 2 * rgb.shape[2]
# out[..., 1] = (out[..., 1] + 1) / 2 * rgb.shape[1]
# out = out.squeeze()
# points_world = self.converter.cam_to_world(out).numpy()

# aim = (points_world[1] + points_world[0]) / 2.0
# angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
# steer = self._turn_controller.step(angle)
# steer = np.clip(steer, -1.0, 1.0)

# desired_speed = np.linalg.norm(np.mean(points_world[:-1] - points_world[1:], 0)) * 2.0
# desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0
# desired_speed *= (1 - abs(angle)) ** 2
