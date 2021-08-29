#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the key configuration parameters for a route-based scenario
"""

import carla
from agents.navigation.local_planner import RoadOption

from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration


class MultiEgoRouteConfiguration(object):

    """
    This clas provides the basic  configuration for a route with multiple ego vehicles
    """

    def __init__(self, route_list=None):
        self.data = route_list

    def parse_xml(self, node):
        """
        Parse route config XML
        """
        self.data = []
        for route in node.iter("route"):
            for waypoint in node.iter("waypoint"):
                x = float(waypoint.attrib.get('x', 0))
                y = float(waypoint.attrib.get('y', 0))
                z = float(waypoint.attrib.get('z', 0))
                c = waypoint.attrib.get('connection', '')
                connection = RoadOption[c.split('.')[1]]

                self.data.append((carla.Location(x, y, z), connection))


class RouteScenarioConfiguration(ScenarioConfiguration):

    """
    Basic configuration of a RouteScenario
    """

    trajectory = None
    scenario_file = None
