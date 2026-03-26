import os
import subprocess
import time
import socket
import gym
import h5py
import pygame
import cv2
from pygame.locals import *
import random

from config import CONFIG

from carla_env.tools.hud import HUD
from carla_env.navigation.controller import VehiclePIDController
from carla_env.navigation.planner import RoadOption, compute_route_waypoints
from carla_env.wrappers import *

import carla
from collections import deque
import itertools

intersection_routes = itertools.cycle(
    [(57, 81), (70, 11), (70, 12), (78, 68), (74, 41), (42, 73), (71, 62), (74, 40), (71, 77), (6, 12), (65, 52),
     (63, 80)])
eval_routes = itertools.cycle(
    [(48, 21), (0, 72), (28, 83), (61, 39), (27, 14), (6, 67), (61, 49), (37, 64), (33, 80), (12, 30), ])

COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_MAGENTA_2 = (255, 140, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_YELLOW_2 = (160, 160, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_3 = (136, 138, 133)
COLOR_ALUMINIUM_5 = (46, 52, 54)


def tint(color, factor):
    r, g, b = color
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    r = min(r, 255)
    g = min(g, 255)
    b = min(b, 255)
    return (r, g, b)


discrete_actions = {
    0: [-1.0, 0.0],
    1: [0.7, -0.5],
    2: [0.7, -0.3],
    3: [0.7, -0.2],
    4: [0.7, -0.1],
    5: [0.7, 0.0],
    6: [0.7, 0.1],
    7: [0.7, 0.2],
    8: [0.7, 0.3],
    9: [0.7, 0.5],
    10: [0.3, -0.7],
    11: [0.3, -0.5],
    12: [0.3, -0.3],
    13: [0.3, -0.2],
    14: [0.3, -0.1],
    15: [0.3, 0.0],
    16: [0.3, 0.1],
    17: [0.3, 0.2],
    18: [0.3, 0.3],
    19: [0.3, 0.5],
    20: [0.3, 0.7],
    21: [0.0, -1.0],
    22: [0.0, -0.6],
    23: [0.0, -0.3],
    24: [0.0, -0.1],
    25: [0.0, 0.0],
    26: [0.0, 0.1],
    27: [0.0, 0.3],
    28: [0.0, 0.6],
    29: [0.0, 1.0],
}

class_blueprint = {
    'car': ['vehicle.tesla.model3',
            'vehicle.audi.tt',
            'vehicle.chevrolet.impala', ]
}


def random_choice_from_blueprint(blueprint):
    all_elements = [item for sublist in blueprint.values() for item in sublist]
    return random.choice(all_elements)


class CarlaRouteEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    @staticmethod
    def _wait_for_carla_server(host, port, timeout=90.0, poll_interval=1.0):
        """Wait until CARLA TCP port is reachable."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=2.0):
                    return True
            except OSError:
                time.sleep(poll_interval)
        return False

    def __init__(self, host="127.0.0.1", port=2000,
                 viewer_res=(1120, 560), obs_res=(80, 120),
                 reward_fn=None,
                 observation_space=None,
                 encode_state_fn=None,
                 fps=15, action_smoothing=0.0,
                 action_space_type="continuous",
                 activate_spectator=True,
                 activate_bev=False,
                 start_carla=False,
                 eval=False,
                 activate_render=True,
                 activate_traffic_flow=False,
                 activate_seg_bev=False,
                 tf_num=20,
                 town='Town02'):

        self.carla_process = None
        if start_carla:
            CARLA_ROOT = "/home/ubuntu/wy/CARLA_0.9.13"
            carla_path = os.path.join(CARLA_ROOT, "CarlaUE4.sh")
            launch_command = [carla_path]
            launch_command += ['-quality_level=Low']
            launch_command += ['-benchmark']
            launch_command += ["-fps=%i" % fps]
            if not activate_render:
                launch_command += ['-RenderOffScreen']
            launch_command += ['-prefernvidia']
            launch_command += [f'-carla-world-port={port}']
            print("Running command:")
            print(" ".join(launch_command))
            self.carla_process = subprocess.Popen(launch_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print("Waiting for CARLA to initialize\n")
            if not self._wait_for_carla_server(host, port, timeout=120.0, poll_interval=1.0):
                raise RuntimeError(
                    f"CARLA server did not become ready within 120s at {host}:{port}. "
                    "Please check CARLA logs and GPU/port availability."
                )

        width, height = viewer_res
        if obs_res is None:
            out_width, out_height = width, height
        else:
            out_width, out_height = obs_res
        self.activate_render = activate_render

        self.num_envs = 1

        # Setup gym environment
        self.action_space_type = action_space_type
        if self.action_space_type == "continuous":
            self.action_space = gym.spaces.Box(np.array([-1, -1]), np.array([1, 1]),
                                               dtype=np.float32)  # steer, throttle
        elif self.action_space_type == "discrete":
            self.action_space = gym.spaces.Discrete(len(discrete_actions))

        self.observation_space = observation_space

        self.fps = fps
        self.action_smoothing = action_smoothing
        self.episode_idx = -2

        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
        self.max_distance = 3000  # m
        self.activate_spectator = activate_spectator
        self.activate_bev = activate_bev
        self.eval = eval
        self.activate_traffic_flow = activate_traffic_flow
        self.train_curriculum = CONFIG.get("train_curriculum", {})
        self.curriculum_phase_override = None
        self.base_tf_num = tf_num
        self.tf_num = tf_num
        self.traffic_flow_vehicles = []
        self.low_speed_timer = 0.0
        self.collision_num = 0
        self.cps = 0  # average collision per timestep
        self.cpm = 0  # average collision per kilometer
        self.collision_speed = 0.0
        self.collision_interval = 0
        self.last_collision_step = 0
        self.collision_deque = deque(maxlen=100)
        self.total_steps = 0
        self.expert_controller = None
        self.sensor_wait_timeout = 5.0

        # bev parameters
        self.use_seg_bev = True if activate_seg_bev else False
        self._width = 192
        self._pixels_per_meter = 5.0
        self._distance_threshold = np.ceil(self._width / self._pixels_per_meter)
        self._scale_bbox = True
        self._history_queue = deque(maxlen=20)
        self._pixels_ev_to_bottom = 40
        self._history_idx = [-16, -11, -6, -1]
        self._scale_mask_col = 1.0
        maps_h5_path = './carla_env/envs/{}.h5'.format(town)
        with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf:
            self._road = np.array(hf['road'], dtype=np.uint8)
            self._lane_marking_all = np.array(hf['lane_marking_all'], dtype=np.uint8)
            self._lane_marking_white_broken = np.array(hf['lane_marking_white_broken'], dtype=np.uint8)
            self._world_offset = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)
            assert np.isclose(self._pixels_per_meter, float(hf.attrs['pixels_per_meter']))

        self.world = None
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(120.0)  # Map loading can take >30s on cold start

            last_err = None
            for _ in range(3):
                try:
                    self.world = World(self.client, town=town)
                    break
                except RuntimeError as e:
                    last_err = e
                    time.sleep(2.0)
            if self.world is None and last_err is not None:
                raise last_err

            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 1 / self.fps
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            self.client.reload_world(False)
            if not self.eval:
                self.world.set_weather(
                    carla.WeatherParameters(cloudiness=100.0, precipitation=0.0, sun_altitude_angle=45.0, )
                )

            self.vehicle = Vehicle(self.world, self.world.map.get_spawn_points()[0],
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e),
                                   is_ego=True)

            if self.activate_render:
                pygame.init()
                pygame.font.init()
                self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.clock = pygame.time.Clock()
                self.hud = HUD(width, height)
                self.hud.set_vehicle(self.vehicle)
                self.world.on_tick(self.hud.on_world_tick)

            seg_settings = {}
            if "seg_camera" in self.observation_space.spaces:
                seg_settings.update({
                    'camera_type': "sensor.camera.semantic_segmentation",
                    'custom_palette': True
                })
            if not self.use_seg_bev:
                self.dashcam = Camera(self.world, out_width, out_height,
                                      transform=sensor_transforms["dashboard"],
                                      attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image(e),
                                      **seg_settings)

            if self.activate_spectator:
                self.camera = Camera(self.world, width, height,
                                     transform=sensor_transforms["spectator"],
                                     attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e))
                self.bev_spectator = Camera(self.world, height // 2, height // 2,
                                            transform=sensor_transforms["birdview0"],
                                            attach_to=self.vehicle,
                                            on_recv_image=lambda e: self._set_bev_spectator_data(e))
            if self.activate_bev:
                self.bev = Camera(self.world, 517, 517,
                                  transform=sensor_transforms["birdview1"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_bev_data(e))

            if self.activate_traffic_flow:
                self.tf_num = tf_num
                self.traffic_manager = self.client.get_trafficmanager(port + 6000)
                self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
                self.traffic_manager.set_synchronous_mode(True)
                self.traffic_manager.set_hybrid_physics_mode(True)
                self.traffic_manager.set_hybrid_physics_radius(50.0)
        except Exception as e:
            self.close()
            raise e
        # Reset env to set initial state
        self.reset()

    def reset(self, is_training=False):
        # Create new route
        self.num_routes_completed = -1
        self.episode_idx += 1
        self.new_route()

        self.terminal_state = False  # Set to True when we want to end episode
        self.success_state = False  # Set to True when we want to end episode.
        self.terminal_reason = "Running..."
        self.collision_state = False
        self._history_queue.clear()
        self.world.world = None
        self.world.world = self.client.get_world()

        self.closed = False  # Set to True when ESC is pressed
        self.extra_info = []  # List of extra info shown on the HUD
        self.observation = self.observation_buffer = None  # Last received observation
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.bev_spectator_data = self.bev_spectator_data_buffer = None
        self.bev_data = self.bev_data_buffer = None
        self.step_count = 0

        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.collision_num = 0
        self.cps = 0
        self.cpm = 0
        self.collision_interval = 0
        self.collision_speed = 0.0
        self.routes_completed = 0.0
        self.low_speed_timer = 0.0
        self.collision = False
        self.action_list = []
        self.prev_steer = 0.0
        self.reward_components = {}
        self.expert_controller = VehiclePIDController(
            self.vehicle,
            args_lateral={'K_P': 1.6, 'K_D': 0.05, 'K_I': 0.2, 'dt': 1.0 / self.fps},
            args_longitudinal={'K_P': 0.8, 'K_D': 0.0, 'K_I': 0.15, 'dt': 1.0 / self.fps},
        )
        self.world.tick()

        time.sleep(0.2)
        obs = self.step(None)[0]
        time.sleep(0.2)

        return obs

    def _get_curriculum_phase(self):
        if self.eval or not self.train_curriculum or not self.train_curriculum.get("enabled", False):
            return None

        phases = self.train_curriculum.get("phases", [])
        if self.curriculum_phase_override is not None:
            if 0 <= int(self.curriculum_phase_override) < len(phases):
                return phases[int(self.curriculum_phase_override)]
            return None

        episode_num = max(self.episode_idx, 0)
        for phase in phases:
            episode_end = phase.get("episode_end", -1)
            if episode_end == -1 or episode_num < episode_end:
                return phase
        return None

    def get_curriculum_phase_index(self):
        if self.eval or not self.train_curriculum or not self.train_curriculum.get("enabled", False):
            return -1
        if self.curriculum_phase_override is not None:
            return int(self.curriculum_phase_override)
        episode_num = max(self.episode_idx, 0)
        for idx, phase in enumerate(self.train_curriculum.get("phases", [])):
            episode_end = phase.get("episode_end", -1)
            if episode_end == -1 or episode_num < episode_end:
                return idx
        return len(self.train_curriculum.get("phases", [])) - 1

    def set_curriculum_phase(self, phase_index):
        self.curriculum_phase_override = None if phase_index is None else int(phase_index)

    @staticmethod
    def _route_length_score(route_length, min_waypoints, max_waypoints):
        if route_length <= 1:
            return float("inf")
        if route_length < min_waypoints:
            return float(min_waypoints - route_length)
        if max_waypoints is not None and route_length > max_waypoints:
            return float(route_length - max_waypoints)
        return 0.0

    def get_expert_action(self):
        """Return a planner/PID-based route-following expert action [steer, throttle_or_brake]."""
        if (
            self.expert_controller is None
            or not hasattr(self, "current_waypoint")
            or not hasattr(self, "next_waypoint")
            or not hasattr(self, "route_waypoints")
        ):
            return np.array([0.0, 0.2], dtype=np.float32)

        speed = self.vehicle.get_speed()
        lookahead = 4
        if speed > 12.0:
            lookahead = 6
        if speed > 20.0:
            lookahead = 8

        target_idx = min(self.current_waypoint_index + lookahead, len(self.route_waypoints) - 1)
        target_waypoint = self.route_waypoints[target_idx][0]

        heading_error = abs(self.vehicle.get_angle(target_waypoint))
        target_speed = 24.0
        if heading_error > np.deg2rad(15.0):
            target_speed = 18.0
        if heading_error > np.deg2rad(30.0):
            target_speed = 12.0

        control = self.expert_controller.run_step(target_speed, target_waypoint)
        steer = float(np.clip(control.steer, -1.0, 1.0))

        if speed > target_speed + 2.0:
            throttle_or_brake = -float(np.clip((speed - target_speed) / 20.0, 0.0, 0.5))
        else:
            throttle_or_brake = float(np.clip(control.throttle, 0.0, 0.75))

        return np.array([steer, throttle_or_brake], dtype=np.float32)

    def _compute_signed_lateral_error(self) -> float:
        if not hasattr(self, "current_waypoint") or not hasattr(self, "next_waypoint"):
            return 0.0

        current_loc = vector(self.current_waypoint.transform.location)[:2]
        next_loc = vector(self.next_waypoint.transform.location)[:2]
        ego_loc = vector(self.vehicle.get_location())[:2]
        lane_vec = next_loc - current_loc
        lane_norm = np.linalg.norm(lane_vec)
        if lane_norm < 1e-6:
            return 0.0
        rel = ego_loc - current_loc
        signed_cross = lane_vec[0] * rel[1] - lane_vec[1] * rel[0]
        return float(signed_cross / lane_norm)

    def _get_front_vehicle_distance(self, max_distance=25.0, lateral_threshold=2.5) -> float:
        ego_transform = self.vehicle.get_transform()
        ego_loc = vector(ego_transform.location)
        yaw = np.deg2rad(ego_transform.rotation.yaw)
        heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)

        min_distance = float("inf")
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == self.vehicle.id:
                continue
            rel = vector(actor.get_location())[:2] - ego_loc[:2]
            longitudinal = float(np.dot(rel, heading))
            lateral = float(abs(heading[0] * rel[1] - heading[1] * rel[0]))
            if longitudinal <= 0.0 or longitudinal > max_distance:
                continue
            if lateral > lateral_threshold:
                continue
            distance = float(np.linalg.norm(rel))
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def get_safety_signals(self, front_scan_distance=25.0, front_lateral_threshold=2.5):
        return {
            "speed": float(self.vehicle.get_speed()),
            "distance_from_center": float(getattr(self, "distance_from_center", 0.0)),
            "signed_lateral_error": self._compute_signed_lateral_error(),
            "front_vehicle_distance": self._get_front_vehicle_distance(
                max_distance=front_scan_distance,
                lateral_threshold=front_lateral_threshold,
            ),
        }

    def _set_terminal_state(self, reason):
        if not self.success_state and not self.terminal_state:
            self.terminal_state = True
            self.terminal_reason = reason

    def _set_success_state(self, reason="Reached destination"):
        if not self.terminal_state and not self.success_state:
            self.success_state = True
            self.terminal_reason = reason

    def new_route(self):
        if self.activate_traffic_flow:
            for bg_veh in list(self.traffic_flow_vehicles):
                if bg_veh.is_alive:
                    bg_veh.destroy()
                    self.traffic_flow_vehicles.remove(bg_veh)

        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        self.vehicle.set_simulate_physics(False)

        if not self.eval:
            curriculum_phase = self._get_curriculum_phase()
            if curriculum_phase is None and self.episode_idx % 2 == 0 and self.num_routes_completed == -1:
                spawn_points_list = [self.world.map.get_spawn_points()[index] for index in next(intersection_routes)]
            elif curriculum_phase is None:
                spawn_points_list = random.sample(self.world.map.get_spawn_points(), 2)
            else:
                spawn_points = self.world.map.get_spawn_points()
                min_waypoints = int(curriculum_phase.get("min_waypoints", 2))
                max_waypoints = curriculum_phase.get("max_waypoints", None)
                if max_waypoints is not None:
                    max_waypoints = int(max_waypoints)
                self.tf_num = int(curriculum_phase.get("tf_num", self.base_tf_num))

                best_candidate = None
                best_score = float("inf")
                for _ in range(200):
                    candidate = random.sample(spawn_points, 2)
                    start_wp, end_wp = [self.world.map.get_waypoint(spawn.location) for spawn in candidate]
                    candidate_route = compute_route_waypoints(self.world.map, start_wp, end_wp, resolution=1.0)
                    route_length = len(candidate_route)
                    score = self._route_length_score(route_length, min_waypoints, max_waypoints)
                    if score < best_score:
                        best_candidate = candidate
                        best_score = score
                    if score == 0.0:
                        break

                spawn_points_list = best_candidate if best_candidate is not None else random.sample(spawn_points, 2)
        else:
            spawn_points_list = [self.world.map.get_spawn_points()[index] for index in next(eval_routes)]

        if self.eval or self._get_curriculum_phase() is None:
            self.tf_num = self.base_tf_num
        route_length = 1
        while route_length <= 1:
            self.start_wp, self.end_wp = [self.world.map.get_waypoint(spawn.location) for spawn in
                                          spawn_points_list]
            self.route_waypoints = compute_route_waypoints(self.world.map, self.start_wp, self.end_wp, resolution=1.0)
            route_length = len(self.route_waypoints)
            if route_length <= 1:
                spawn_points_list = random.sample(self.world.map.get_spawn_points(), 2)

        self.distance_from_center_history = deque(maxlen=30)

        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        self.vehicle.set_transform(self.start_wp.transform)
        time.sleep(0.2)
        self.vehicle.set_simulate_physics(True)

        if self.activate_traffic_flow:
            spawn_points = self.world.get_map().get_spawn_points()
            number_of_vehicles = min(len(spawn_points), self.tf_num)  # Adjust the number of vehicles as needed

            for _ in range(number_of_vehicles):
                blueprint = random_choice_from_blueprint(class_blueprint)
                blueprint = self.world.get_blueprint_library().find(blueprint)
                spawn_point = random.choice(spawn_points)
                bg_veh = self.world.try_spawn_actor(blueprint, spawn_point)

                if bg_veh:
                    bg_veh.set_autopilot(True, self.traffic_manager.get_port())
                    self.traffic_flow_vehicles.append(bg_veh)

    def close(self):
        if self.carla_process:
            self.carla_process.terminate()
            try:
                self.carla_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.carla_process.kill()
                self.carla_process.wait(timeout=5)
            finally:
                self.carla_process = None
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True
        if self.world is not None:
            actors = self.world.get_actors().filter('vehicle.*')
            for actor in actors:
                actor.destroy()

    def render(self, mode="human"):
        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        elif mode == "state_pixels":
            return self.observation_render

        self.clock.tick()
        self.hud.tick(self.world, self.clock)

        if self.current_road_maneuver == RoadOption.LANEFOLLOW:
            maneuver = "Follow Lane"
        elif self.current_road_maneuver == RoadOption.LEFT:
            maneuver = "Left"
        elif self.current_road_maneuver == RoadOption.RIGHT:
            maneuver = "Right"
        elif self.current_road_maneuver == RoadOption.STRAIGHT:
            maneuver = "Straight"
        elif self.current_road_maneuver == RoadOption.CHANGELANELEFT:
            maneuver = "Change Lane Left"
        elif self.current_road_maneuver == RoadOption.CHANGELANERIGHT:
            maneuver = "Change Lane Right"
        else:
            maneuver = "INVALID"

        self.extra_info.extend([
            "Episode {}".format(self.episode_idx),
            "",
            "Maneuver:  % 17s" % maneuver,
            "Throttle:            %7.2f" % self.vehicle.control.throttle,
            "Brake:               %7.2f" % self.vehicle.control.brake,
            "Steer:               %7.2f" % self.vehicle.control.steer,
            "Routes completed:    % 7.2f" % self.routes_completed,
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (self.speed_accum / self.step_count),
        ])

        if self.activate_spectator:
            self.viewer_image = self._draw_path(self.camera, self.viewer_image)
            self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

            new_size = (self.display.get_size()[1] // 2, self.display.get_size()[1] // 2)
            pos_bev = (self.display.get_size()[0] - self.display.get_size()[1] // 2, self.display.get_size()[1] // 2)
            bev_surface = pygame.surfarray.make_surface(self.bev_spectator_data.swapaxes(0, 1))
            scaled_surface = pygame.transform.scale(bev_surface, new_size)
            self.display.blit(scaled_surface, pos_bev)

            pos_obs = (self.display.get_size()[0] - self.display.get_size()[1] // 2, 0)
            obs_surface = pygame.surfarray.make_surface(self.observation_render.swapaxes(0, 1))
            scaled_surface = pygame.transform.scale(obs_surface, new_size)
            self.display.blit(scaled_surface, pos_obs)

        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []

        pygame.display.flip()

    def step(self, action):
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")
        if action is not None:
            if self.action_space_type == "continuous":
                steer, throttle = [float(a) for a in action]
            elif self.action_space_type == "discrete":
                throttle, steer = discrete_actions[int(action)]

            self.prev_steer = float(self.vehicle.control.steer)
            if self.action_smoothing > 0.0:
                self.vehicle.control.steer = smooth_action(self.vehicle.control.steer, steer, self.action_smoothing)
            else:
                self.vehicle.control.steer = steer
            if throttle >= 0:
                self.vehicle.control.throttle = throttle
                self.vehicle.control.brake = 0
            else:
                self.vehicle.control.throttle = 0
                self.vehicle.control.brake = -throttle
            self.action_list.append(self.vehicle.control.steer)
        self.world.tick()

        if self.use_seg_bev:
            self.observation_render = self._get_observation_seg_bev()['rendered']
            self.observation = self._get_observation_seg_bev()['masks']
        else:
            self.observation = self._get_observation()
            self.observation_render = self._get_observation_seg_bev()['rendered']
        if self.activate_spectator:
            self.viewer_image = self._get_viewer_image()
            self.bev_spectator_data = self._get_bev_spectator_data()

        if self.activate_bev:
            self.bev_data = self._get_bev_data()

        transform = self.vehicle.get_transform()

        self.prev_waypoint_index = self.current_waypoint_index
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0:
                waypoint_index += 1
            else:
                break
        self.current_waypoint_index = min(waypoint_index, len(self.route_waypoints) - 1)

        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
                (self.current_waypoint_index + 1) % len(self.route_waypoints)]
        else:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
                self.current_waypoint_index % len(self.route_waypoints)]

        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[
            self.current_waypoint_index % len(self.route_waypoints)]
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(
            self.route_waypoints)

        if action is not None and self.current_waypoint_index >= len(self.route_waypoints) - 1:
            self._set_success_state("Reached destination")

        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        if action is not None:
            self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        self.speed_accum += self.vehicle.get_speed()
        
        # Low speed timeout: terminate if vehicle is stuck (speed < 3 km/h for too long)
        # 100 frames @ 15 fps ~= 6.7 seconds
        # Only applies during training, not evaluation
        current_speed = self.vehicle.get_speed()
        if current_speed < 3.0:  # Less than 3 km/h
            self.low_speed_timer += 1
            if self.low_speed_timer >= 100 and not self.eval:
                self._set_terminal_state("Vehicle stopped")
        else:
            self.low_speed_timer = 0  # Reset timer when moving

        if not self.eval:
            if self.distance_from_center > CONFIG.reward_params.max_distance:
                self._set_terminal_state("Off-track")
            if CONFIG.reward_params.max_speed > 0 and current_speed > CONFIG.reward_params.max_speed:
                self._set_terminal_state("Too fast")

        self.distance_from_center_history.append(self.distance_from_center)

        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward

        encoded_state = self.encode_state_fn(self)
        self.step_count += 1
        self.total_steps += 1

        if self.activate_render:
            pygame.event.pump()
            if pygame.key.get_pressed()[K_ESCAPE]:
                self.close()
                self._set_terminal_state("Closed")
            self.extra_info.extend([
                self.terminal_reason,
                ""
            ])
            self.render()

        max_distance = CONFIG.reward_params.max_distance
        max_std_center_lane = CONFIG.reward_params.max_std_center_lane
        max_angle_center_lane = CONFIG.reward_params.max_angle_center_lane
        centering_factor = max(1.0 - self.distance_from_center / max_distance, 0.0)

        angle = self.vehicle.get_angle(self.current_waypoint)
        angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)

        std = np.std(self.distance_from_center_history)
        distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)

        if self.terminal_state or self.success_state:
            if self.collision_state:
                self.collision_num += 1
                self.collision_deque.append(1)
                self.cps = 1 / self.step_count
                self.cpm = 1 / self.distance_traveled * 1000
                self.collision_interval = self.total_steps - self.last_collision_step
                self.last_collision_step = self.total_steps
                self.collision_speed = self.vehicle.get_speed()
            else:
                self.collision_deque.append(0)

        info = {
            "closed": self.closed,
            'total_reward': self.total_reward,
            'routes_completed': self.routes_completed,
            'total_distance': self.distance_traveled,
            'avg_center_dev': (self.center_lane_deviation / self.step_count),
            'avg_speed': (self.speed_accum / self.step_count),
            'mean_reward': (self.total_reward / self.step_count),
            'ground_truth_reward': self.last_reward,
            'terminal_reason': self.terminal_reason,
            'terminal_state': self.terminal_state,
            'success_state': self.success_state,
            'render_array': self.bev_data,
            "centering_factor": centering_factor,
            "angle_factor": angle_factor,
            "distance_std_factor": distance_std_factor,
            "speed": self.vehicle.get_speed(),
            "collision_num": self.collision_num,
            "collision_rate": sum(self.collision_deque) / len(self.collision_deque) if self.collision_deque else 0.0,
            "episode_length": self.step_count,
            "collision_state": self.collision_state,
            "curriculum_phase": self.get_curriculum_phase_index(),
            "reward_components": dict(getattr(self, "reward_components", {})),
        }
        for key, value in getattr(self, "reward_components", {}).items():
            info[f"reward_{key}"] = value

        if self.terminal_state or self.success_state:
            if self.collision_state:
                info.update({"CPS": self.cps,
                             "CPM": self.cpm,
                             "collision_interval": self.collision_interval,
                             "collision_speed": self.collision_speed,
                             })
        return encoded_state, self.last_reward, self.terminal_state or self.success_state, info

    def _draw_path_server(self, life_time=60.0, skip=0):
        """
            Draw a connected path from start of route to end.
            Green node = start
            Red node   = point along path
            Blue node  = destination
        """
        for i in range(0, len(self.route_waypoints) - 1, skip + 1):
            z = 30.25
            w0 = self.route_waypoints[i][0]
            w1 = self.route_waypoints[i + 1][0]
            self.world.debug.draw_line(
                w0.transform.location + carla.Location(z=z),
                w1.transform.location + carla.Location(z=z),
                thickness=0.1, color=carla.Color(255, 0, 0),
                life_time=life_time, persistent_lines=False)
            self.world.debug.draw_point(
                w0.transform.location + carla.Location(z=z), 0.1,
                carla.Color(0, 255, 0) if i == 0 else carla.Color(255, 0, 0),
                life_time, False)
        self.world.debug.draw_point(
            self.route_waypoints[-1][0].transform.location + carla.Location(z=z), 0.1,
            carla.Color(0, 0, 255),
            life_time, False)

    def _draw_path(self, camera, image):
        """
            Draw a connected path from start of route to end using homography.
        """
        vehicle_vector = vector(self.vehicle.get_transform().location)
        # Get the world to camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # Get the attributes from the camera
        image_w = int(camera.actor.attributes['image_size_x'])
        image_h = int(camera.actor.attributes['image_size_y'])
        fov = float(camera.actor.attributes['fov'])
        for i in range(self.current_waypoint_index, len(self.route_waypoints)):
            waypoint_location = self.route_waypoints[i][0].transform.location + carla.Location(z=1.25)
            waypoint_vector = vector(waypoint_location)
            if not (2 < abs(np.linalg.norm(vehicle_vector - waypoint_vector)) < 50):
                continue
            # Calculate the camera projection matrix to project from 3D -> 2D
            K = build_projection_matrix(image_w, image_h, fov)
            x, y = get_image_point(waypoint_location, K, world_2_camera)
            if i == len(self.route_waypoints) - 1:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            # Ensure x and y are Python native int, not numpy types
            center_x = int(round(float(x)))
            center_y = int(round(float(y)))
            image = cv2.circle(image, (center_x, center_y), radius=3, color=color, thickness=-1)
        return image

    def _get_observation(self):
        start = time.time()
        while self.observation_buffer is None:
            if time.time() - start > self.sensor_wait_timeout:
                raise TimeoutError("Timed out waiting for observation_buffer")
            time.sleep(0.001)
        obs = self.observation_buffer.copy()
        self.observation_buffer = None
        return obs

    def _get_viewer_image(self):
        start = time.time()
        while self.viewer_image_buffer is None:
            if time.time() - start > self.sensor_wait_timeout:
                raise TimeoutError("Timed out waiting for viewer_image_buffer")
            time.sleep(0.001)
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _get_bev_spectator_data(self):
        start = time.time()
        while self.bev_spectator_data_buffer is None:
            if time.time() - start > self.sensor_wait_timeout:
                raise TimeoutError("Timed out waiting for bev_spectator_data_buffer")
            time.sleep(0.001)
        image = self.bev_spectator_data_buffer.copy()
        self.bev_spectator_data_buffer = None
        return image

    def _get_bev_data(self):
        start = time.time()
        while self.bev_data_buffer is None:
            if time.time() - start > self.sensor_wait_timeout:
                raise TimeoutError("Timed out waiting for bev_data_buffer")
            time.sleep(0.001)
        image = self.bev_data_buffer.copy()
        self.bev_data_buffer = None
        return image

    def _on_collision(self, event):
        if get_actor_display_name(event.other_actor) != "Road":
            self._set_terminal_state("Collision with {}".format(event.other_actor.type_id))
            self.collision_state = True
            print("0| Terminal:  Collision with {}".format(event.other_actor.type_id))
        if self.activate_render:
            self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

    def _on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        if self.activate_render:
            self.hud.notification("Crossed line %s" % " and ".join(text))

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    def _set_bev_spectator_data(self, image):
        self.bev_spectator_data_buffer = image

    def _set_bev_data(self, image):
        self.bev_data_buffer = image

    def _get_observation_seg_bev(self):
        ev_transform = self.vehicle.get_transform()
        ev_loc = ev_transform.location
        ev_rot = ev_transform.rotation
        ev_bbox = self.vehicle.bounding_box

        def is_within_distance(w):
            c_distance = abs(ev_loc.x - w.location.x) < self._distance_threshold \
                         and abs(ev_loc.y - w.location.y) < self._distance_threshold \
                         and abs(ev_loc.z - w.location.z) < 8.0
            c_ev = abs(ev_loc.x - w.location.x) < 1.0 and abs(ev_loc.y - w.location.y) < 1.0
            return c_distance and (not c_ev)

        vehicle_bbox_list = self.world.get_level_bbs(carla.CityObjectLabel.Vehicles)
        walker_bbox_list = self.world.get_level_bbs(carla.CityObjectLabel.Pedestrians)
        if self._scale_bbox:
            vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance, 1.0)
            walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance, 2.0)
        else:
            vehicles = self._get_surrounding_actors(vehicle_bbox_list, is_within_distance)
            walkers = self._get_surrounding_actors(walker_bbox_list, is_within_distance)

        self._history_queue.append((vehicles, walkers))

        M_warp = self._get_warp_transform(ev_loc, ev_rot)

        # objects with history
        vehicle_masks, walker_masks = self._get_history_masks(M_warp)

        # road_mask, lane_mask
        road_mask = cv2.warpAffine(self._road, M_warp, (self._width, self._width)).astype(bool)
        lane_mask_all = cv2.warpAffine(self._lane_marking_all, M_warp, (self._width, self._width)).astype(bool)
        lane_mask_broken = cv2.warpAffine(self._lane_marking_white_broken, M_warp,
                                          (self._width, self._width)).astype(bool)

        # ev_mask
        ev_mask = self._get_mask_from_actor_list([(ev_transform, ev_bbox.location, ev_bbox.extent)], M_warp)

        # render
        image = np.zeros([self._width, self._width, 3], dtype=np.uint8)
        image[road_mask] = COLOR_ALUMINIUM_5
        # image[route_mask] = COLOR_ALUMINIUM_3
        image[lane_mask_all] = COLOR_MAGENTA
        image[lane_mask_broken] = COLOR_MAGENTA_2

        h_len = len(self._history_idx) - 1

        for i, mask in enumerate(vehicle_masks):
            image[mask] = tint(COLOR_BLUE, (h_len - i) * 0.2)
        for i, mask in enumerate(walker_masks):
            image[mask] = tint(COLOR_CYAN, (h_len - i) * 0.2)

        image[ev_mask] = COLOR_WHITE

        # masks
        c_road = road_mask * 255
        c_lane = lane_mask_all * 255
        c_lane[lane_mask_broken] = 120

        # masks with history
        c_vehicle_history = [m * 255 for m in vehicle_masks]

        masks = np.stack((c_road, c_lane, *c_vehicle_history), axis=2)

        obs_dict = {'rendered': image, 'masks': masks}

        return obs_dict

    @staticmethod
    def _get_surrounding_actors(bbox_list, criterium, scale=None):
        actors = []
        for bbox in bbox_list:
            is_within_distance = criterium(bbox)
            if is_within_distance:
                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bbox.extent)
                if scale is not None:
                    bb_ext = bb_ext * scale
                    bb_ext.x = max(bb_ext.x, 0.8)
                    bb_ext.y = max(bb_ext.y, 0.8)

                actors.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
        return actors

    def _get_warp_transform(self, ev_loc, ev_rot):
        ev_loc_in_px = self._world_to_pixel(ev_loc)
        yaw = np.deg2rad(ev_rot.yaw)

        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        right_vec = np.array([np.cos(yaw + 0.5 * np.pi), np.sin(yaw + 0.5 * np.pi)])

        bottom_left = ev_loc_in_px - self._pixels_ev_to_bottom * forward_vec - (0.5 * self._width) * right_vec
        top_left = ev_loc_in_px + (self._width - self._pixels_ev_to_bottom) * forward_vec - (
            0.5 * self._width) * right_vec
        top_right = ev_loc_in_px + (self._width - self._pixels_ev_to_bottom) * forward_vec + (
            0.5 * self._width) * right_vec

        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.array([[0, self._width - 1],
                            [0, 0],
                            [self._width - 1, 0]], dtype=np.float32)
        return cv2.getAffineTransform(src_pts, dst_pts)

    def _world_to_pixel(self, location, projective=False):
        """Converts the world coordinates to pixel coordinates"""
        x = self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self._pixels_per_meter * (location.y - self._world_offset[1])

        if projective:
            p = np.array([x, y, 1], dtype=np.float32)
        else:
            p = np.array([x, y], dtype=np.float32)
        return p

    def _get_history_masks(self, M_warp):
        qsize = len(self._history_queue)
        vehicle_masks, walker_masks, tl_green_masks, tl_yellow_masks, tl_red_masks, stop_masks = [], [], [], [], [], []
        for idx in self._history_idx:
            idx = max(idx, -1 * qsize)

            vehicles, walkers = self._history_queue[idx]

            vehicle_masks.append(self._get_mask_from_actor_list(vehicles, M_warp))
            walker_masks.append(self._get_mask_from_actor_list(walkers, M_warp))

        return vehicle_masks, walker_masks

    def _get_mask_from_actor_list(self, actor_list, M_warp):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        # for actor_transform, bb_loc, bb_ext in actor_list:
        for data in actor_list:
            if len(data) == 12:
                loc_x, loc_y, loc_z, pitch, yaw, roll, bb_loc_x, bb_loc_y, bb_loc_z, bb_ext_x, bb_ext_y, bb_ext_z = data
                actor_transform = carla.Transform(
                    carla.Location(x=loc_x, y=loc_y, z=loc_z),
                    carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
                )
                bb_loc = carla.Location(x=bb_loc_x, y=bb_loc_y, z=bb_loc_z)
                bb_ext = carla.Vector3D(x=bb_ext_x, y=bb_ext_y, z=bb_ext_z)
            else:
                actor_transform, bb_loc, bb_ext = data
            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=0),
                       carla.Location(x=bb_ext.x, y=bb_ext.y),
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]
            corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])
            corners_warped = cv2.transform(corners_in_pixel, M_warp)

            cv2.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
        return mask.astype(bool)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
