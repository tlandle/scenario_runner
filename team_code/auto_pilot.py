import os
import time
import datetime
import pathlib

import numpy as np
import cv2
import carla

from PIL import Image, ImageDraw

from team_code.carla_project.src.common import CONVERTER, COLOR
from team_code.map_agent import MapAgent
from team_code.pid_controller import PIDController


HAS_DISPLAY = True
DEBUG = False
WEATHERS = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.ClearSunset,

        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.CloudySunset,

        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetSunset,

        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.MidRainSunset,

        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.WetCloudySunset,

        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.HardRainSunset,

        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.SoftRainSunset,
]


def get_entry_point():
    return 'DebugAutoPilot'


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_angle(u, orientation):
    v = np.array([
        np.cos(np.radians(orientation)),
        np.sin(np.radians(orientation))])

    return np.sign(np.cross(u, v)) * np.degrees(np.arccos(np.dot(u, v) / (np.linalg.norm(u) + 1e-3)))


def get_dot(target_orientation, current_orientation):
    u = np.array([
        np.cos(np.radians(target_orientation)),
        np.sin(np.radians(target_orientation))])
    v = np.array([
        np.cos(np.radians(current_orientation)),
        np.sin(np.radians(current_orientation))])

    return np.dot(u, v)


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)

    return collides, p1 + x[0] * v1


class AutoPilot(MapAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self.save_path = None

        if path_to_conf_file:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

            print(string)

            self.save_path = pathlib.Path(path_to_conf_file) / string
            self.save_path.mkdir(exist_ok=False)

            (self.save_path / 'rgb').mkdir()
            (self.save_path / 'rgb_left').mkdir()
            (self.save_path / 'rgb_right').mkdir()
            (self.save_path / 'topdown').mkdir()
            (self.save_path / 'measurements').mkdir()

    def _init(self):
        super()._init()

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=0.5, n=40)

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._waypoint_planner.mean) * self._waypoint_planner.scale

        return gps

    def _get_speed(self, tick_data):
        return tick_data['speed']

    def _get_orientation(self, tick_data):
        return tick_data['compass']

    def _get_angle_to(self, pos, theta, target):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))

        return angle

    def _get_control(self, target, far_target, tick_data, _draw):
        pos = self._get_position(tick_data)
        theta = self._get_orientation(tick_data)
        speed = self._get_speed(tick_data)

        # Steering.
        angle_unnorm = self._get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90
        # angle = angle if abs(angle_unnorm) > 1.0 else 0.0

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        target_speed = 4 if should_slow else 7.0

        brake = self._should_brake()
        target_speed = target_speed if not brake else 0.0

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        if brake:
            steer *= 0.5
            throttle = 0.0

        _draw.text((5, 90), 'Speed: %.3f' % speed)
        _draw.text((5, 110), 'Target: %.3f' % target_speed)
        _draw.text((5, 130), 'Angle: %.3f' % angle_unnorm)
        _draw.text((5, 150), 'Angle Far: %.3f' % angle_far_unnorm)

        return steer, throttle, brake, target_speed

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        if self.step % 100 == 0:
            index = (self.step // 100) % len(WEATHERS)
            self._world.set_weather(WEATHERS[index])

        data = self.tick(input_data)
        topdown = data['topdown']
        rgb = np.hstack((data['rgb_left'], data['rgb'], data['rgb_right']))

        gps = self._get_position(data)

        near_node, near_command = self._waypoint_planner.run_step(gps)
        far_node, far_command = self._command_planner.run_step(gps)

        _topdown = Image.fromarray(COLOR[CONVERTER[topdown]])
        _rgb = Image.fromarray(rgb)
        _draw = ImageDraw.Draw(_topdown)

        _topdown.thumbnail((256, 256))
        _rgb = _rgb.resize((int(256 / _rgb.size[1] * _rgb.size[0]), 256))

        _combined = Image.fromarray(np.hstack((_rgb, _topdown)))
        _draw = ImageDraw.Draw(_combined)

        steer, throttle, brake, target_speed = self._get_control(near_node, far_node, data, _draw)

        _draw.text((5, 10), 'FPS: %.3f' % (self.step / (time.time() - self.wall_start)))
        _draw.text((5, 30), 'Steer: %.3f' % steer)
        _draw.text((5, 50), 'Throttle: %.3f' % throttle)
        _draw.text((5, 70), 'Brake: %s' % brake)

        if HAS_DISPLAY:
            cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

        control = carla.VehicleControl()
        control.steer = steer + 1e-2 * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake)

        if self.step % 10 == 0:
            self.save(far_node, near_command, steer, throttle, brake, target_speed, data)

        return control

    def save(self, far_node, near_command, steer, throttle, brake, target_speed, tick_data):
        frame = self.step // 10

        pos = self._get_position(tick_data)
        theta = self._get_orientation(tick_data)
        speed = self._get_speed(tick_data)

        data = {
                'x': pos[0],
                'y': pos[1],
                'theta': theta,
                'speed': speed,
                'target_speed': target_speed,
                'x_command': far_node[0],
                'y_command': far_node[1],
                'command': near_command.value,
                'steer': steer,
                'throttle': throttle,
                'brake': brake,
                }

        (self.save_path / 'measurements' / ('%04d.json' % frame)).write_text(str(data))

        Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))
        Image.fromarray(tick_data['rgb_left']).save(self.save_path / 'rgb_left' / ('%04d.png' % frame))
        Image.fromarray(tick_data['rgb_right']).save(self.save_path / 'rgb_right' / ('%04d.png' % frame))
        Image.fromarray(tick_data['topdown']).save(self.save_path / 'topdown' / ('%04d.png' % frame))

    def _should_brake(self):
        actors = self._world.get_actors()

        vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        light = self._is_light_red(actors.filter('*traffic_light*'))
        walker = self._is_walker_hazard(actors.filter('*walker*'))

        return any(x is not None for x in [vehicle, light, walker])

    def _draw_line(self, p, v, z, color=(255, 0, 0)):
        if not DEBUG:
            return

        p1 = _location(p[0], p[1], z)
        p2 = _location(p[0]+v[0], p[1]+v[1], z)
        color = carla.Color(*color)

        self._world.debug.draw_line(p1, p2, 0.25, color, 0.01)

    def _is_light_red(self, lights_list):
        if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
            return self._vehicle.get_traffic_light()

        return None

    def _is_walker_hazard(self, walkers_list):
        z = self._vehicle.get_location().z
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

        self._draw_line(p1, v1, z+2.5, (0, 0, 255))

        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat

            self._draw_line(p2, v2, z+2.5)

            collides, collision_point = get_collision(p1, v1, p2, v2)

            if collides:
                return walker

        return None

    def _is_vehicle_hazard(self, vehicle_list):
        z = self._vehicle.get_location().z

        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._vehicle.get_location())
        s1 = max(7.5, 2.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity())))
        v1_hat = o1
        v1 = s1 * v1_hat

        self._draw_line(p1, v1, z+2.5, (255, 0, 0))

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            self._draw_line(p2, v2, z+2.5, (255, 0, 0))

            if np.degrees(np.arccos(o1.dot(o2))) > 60.0:
                continue
            elif np.degrees(np.arccos(v1_hat.dot(p2_p1_hat))) > 30.0:
                continue
            elif distance > s1:
                continue

            return target_vehicle

        return None


class DebugAutoPilot(AutoPilot):
    def run_step(self, *args, **kwargs):
        import traceback

        try:
            return super().run_step(*args, **kwargs)
        except Exception:
            traceback.print_exc()
