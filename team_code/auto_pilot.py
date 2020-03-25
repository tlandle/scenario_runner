import sys; sys.path.append('/home/bradyzhou/code/')

import time
import datetime
import pathlib

import numpy as np
import cv2
import carla

from PIL import Image, ImageDraw

from carla_random.src.common import CONVERTER, COLOR

from team_code.map_agent import MapAgent
from team_code.common import PIDController


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
    return 'AutoPilot'


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


def _numpy(carla_vector):
    return np.float32([carla_vector.x, carla_vector.y])


class AutoPilot(MapAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        if not path_to_conf_file:
            self.save_path = None
        else:
            now = datetime.datetime.now()
            string = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

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

        self._turn_controller = PIDController(K_P=1.0, K_I=0.5, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=0.5, K_I=0.75, K_D=0.4, n=40)

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._waypoint_planner.mean) * self._waypoint_planner.scale

        return gps

    def _get_speed(self, tick_data):
        return tick_data['speed']

    def _get_target_speed(self, command, tick_data):
        if command.name != 'LANEFOLLOW':
            return 5.0

        return self._vehicle.get_speed_limit() / 3.6 * 0.9

    def _get_orientation(self, tick_data):
        return tick_data['compass']

    def _get_steer(self, target, command, tick_data, _draw):
        pos = self._get_position(tick_data)
        theta = self._get_orientation(tick_data)
        speed = self._get_speed(tick_data)
        target_speed = self._get_target_speed(command, tick_data)

        # Steering.
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])
        aim = R.T.dot(target - pos)
        angle = (-np.degrees(np.arctan2(-aim[1], aim[0]))) / 90
        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        # Acceleration.
        delta = np.clip(target_speed - speed, 0.0, 1.0)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        brake = self._get_brake()

        if brake:
            throttle = 0.0

        _draw.text((5, 90), 'Speed: %.3f' % speed)
        _draw.text((5, 110), 'Target: %.3f' % target_speed)
        _draw.text((5, 130), 'Angle: %.3f' % angle)

        return steer, throttle, brake

    def save(self, far_node, near_command, steer, throttle, brake, tick_data):
        frame = self.step // 10

        pos = self._get_position(tick_data)
        theta = self._get_orientation(tick_data)
        speed = self._get_speed(tick_data)

        data = {
                'x': pos[0],
                'y': pos[1],
                'theta': theta,
                'speed': speed,
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

        steer, throttle, brake = self._get_steer(near_node, near_command, data, _draw)

        _draw.text((5, 10), 'FPS: %.3f' % (self.step / (time.time() - self.wall_start)))
        _draw.text((5, 30), 'Steer: %.3f' % steer)
        _draw.text((5, 50), 'Throttle: %.3f' % throttle)
        _draw.text((5, 70), 'Brake: %s' % brake)

        if HAS_DISPLAY:
            cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake

        if self.step % 10 == 0:
            self.save(far_node, near_command, steer, throttle, brake, data)

        return control

    def _get_brake(self):
        actors = self._world.get_actors()

        blocking_vehicle, vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        blocking_light, traffic_light = self._is_light_red(actors.filter('*traffic_light*'))
        blocking_walker, walker = self._is_walker_hazard(actors.filter('*walker*'))

        return blocking_vehicle or blocking_light or blocking_walker

    def _is_light_red(self, lights_list):
        if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
            return True, self._vehicle.get_traffic_light()

        return False, None

    def _is_walker_hazard(self, walkers_list):
        p1 = _numpy(self._vehicle.get_location())
        v1 = _numpy(self._vehicle.get_velocity())
        v1_hat = v1 / (np.linalg.norm(v1) + 1e-3)
        v1 = max(15.0, 3.0 * np.linalg.norm(v1)) * v1_hat

        for walker in walkers_list:
            v2 = _numpy(walker.get_velocity())
            v2_hat = v2 / (np.linalg.norm(v2) + 1e-3)
            p2 = _numpy(walker.get_location()) - 1.0 * v2_hat

            collides, collision_point = get_collision(p1, v1, p2, 6.0 * v2)

            if DEBUG:
                self._world.debug.draw_line(
                        carla.Location(x=float(p2[0]), y=float(p2[1]), z=float(walker.get_location().z+2.5)),
                        carla.Location(x=float(p2[0]+v2[0]), y=float(p2[1]+v2[1]), z=float(walker.get_location().z+2.5)),
                        life_time=0.01, thickness=0.5)

            if collides:
                return True, walker

        return False, None

    def _is_vehicle_hazard(self, vehicle_list):
        p1 = _numpy(self._vehicle.get_location())
        speed = np.linalg.norm(_numpy(self._vehicle.get_velocity()))
        v1 = 2.0 * _numpy(self._vehicle.get_velocity())
        o1 = self._vehicle.get_transform().rotation.yaw

        if DEBUG:
            self._world.debug.draw_line(
                    carla.Location(x=float(p1[0]), y=float(p1[1]), z=float(self._vehicle.get_location().z+2.5)),
                    carla.Location(x=float(p1[0]+v1[0]), y=float(p1[1]+v1[1]), z=float(self._vehicle.get_location().z+2.5)),
                    life_time=0.01, thickness=0.5)

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            p2 = _numpy(target_vehicle.get_location())
            v2 = 2.0 * _numpy(target_vehicle.get_velocity())
            o2 = target_vehicle.get_transform().rotation.yaw

            collides, _ = get_collision(p1, v1, p2, v2)
            distance = np.linalg.norm(p1 - p2)
            angle = get_angle(p2 - p1, o1)
            dot = get_dot(o1, o2)

            if DEBUG:
                self._world.debug.draw_line(
                        carla.Location(x=float(p2[0]), y=float(p2[1]), z=float(target_vehicle.get_location().z+2.5)),
                        carla.Location(x=float(p2[0]+v2[0]), y=float(p2[1]+v2[1]), z=float(target_vehicle.get_location().z+2.5)),
                        life_time=0.01, thickness=0.5)

            if collides:
                return True, target_vehicle
            elif abs(angle) <= 15 and dot >= 0 and (distance < 9.0 or distance < 2.0 * speed):
                return True, target_vehicle

        return False, None
