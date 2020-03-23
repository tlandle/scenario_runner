# import sys; sys.path.append('/home/bradyzhou/code/')
import time

import numpy as np
import cv2
import carla
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

from srunner.scenariomanager.carla_data_provider import CarlaActorPool
from leaderboard.autoagents import autonomous_agent
from leaderboard.utils.route_manipulation import _location_to_gps, _get_latlon_ref


def get_entry_point():
    return 'ExampleAgent'


class ExampleAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.initialized = False

    def sensors(self):
        return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 256, 'height': 144, 'fov': 90,
                    'id': 'rgb'
                    },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    }
                ]

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        self._global_plan_gps_NO_DOWNSAMPLE = global_plan_gps

    def _init(self):
        self.initialized = True
        self.step = 0
        self.wall_start = time.time()
        self._vehicle = CarlaActorPool.get_hero_actor()

    def tick(self, input_data):
        self.step += 1

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        sensor_gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        hack_gps = _location_to_gps(*_get_latlon_ref(self._vehicle.get_world()), self._vehicle.get_location())
        hack_gps = np.array([hack_gps['lat'], hack_gps['lon']])

        plt.ion()
        plt.clf()
        plt.plot(*hack_gps, 'ro', label='hack gps')
        plt.plot(*sensor_gps, 'bo', label='sensor gps')

        for pos, _ in self._global_plan_gps_NO_DOWNSAMPLE[:100]:
            plt.plot(pos['lat'], pos['lon'], 'c.')

        plt.legend()
        plt.axis('equal')
        plt.pause(1e-5)

        return {
                'rgb': rgb,
                'sensor_gps': sensor_gps,
                'hack_gps': hack_gps,
                'speed': speed,
                'compass': compass
                }

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)
        rgb = tick_data['rgb']

        control = carla.VehicleControl()
        control.steer = -0.1
        control.throttle = 0.75
        control.brake = 0.0

        _rgb = Image.fromarray(rgb)
        _draw = ImageDraw.Draw(_rgb)
        _draw.text((5, 10), 'FPS: %.3f' % (self.step / (time.time() - self.wall_start)))

        cv2.imshow('rgb', cv2.cvtColor(np.array(_rgb), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

        return control
