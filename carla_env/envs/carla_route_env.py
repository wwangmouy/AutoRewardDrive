import os
import subprocess
import time
import gym
import h5py
import pygame
import cv2
from pygame.locals import *
import random

from config import CONFIG

from carla_env.tools.hud import HUD
from carla_env.navigation.planner import RoadOption, compute_route_waypoints
from carla_env.wrappers import *

import carla
from collections import deque
import itertools
from safe_field import FDPF
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend  
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

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
            CARLA_ROOT = "/home/wy/CARLA_0.9.13"
            carla_path = os.path.join(CARLA_ROOT, "CarlaUE4.sh")
            launch_command = [carla_path]
            launch_command += ['-quality_level=Low']
            launch_command += ['-benchmark']
            launch_command += ["-fps=%i" % fps]
            launch_command += ['-RenderOffScreen']
            launch_command += ['-prefernvidia']
            launch_command += [f'-carla-world-port={port}']
            print("Running command:")
            print(" ".join(launch_command))
            self.carla_process = subprocess.Popen(launch_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print("Waiting for CARLA to initialize\n")

            time.sleep(5)

        width, height = viewer_res
        if obs_res is None:
            out_width, out_height = width, height
        else:
            out_width, out_height = obs_res
        self.activate_render = activate_render

        # Increase window height to accommodate 3 panels vertically (BEV, RGB, FDPF)
        # Original height: 560, New height: 840 (to fit 3x 280px panels)
        display_width = width
        display_height = max(height, 840)  # Ensure at least 840px height for 3 panels

        self.num_envs = 1

        # Initialize FDPF
        self.fdpf = FDPF()
        self.fdpf.vehicle_width = 2.0 # Approx width
        self.fdpf.lane_width = 3.5 # Approx lane width

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
            self.client.set_timeout(5.0)
            self.world = World(self.client, town=town)

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
                self.display = pygame.display.set_mode((display_width, display_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.clock = pygame.time.Clock()
                self.hud = HUD(display_width, display_height)
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
        self.routes_completed = 0.0
        self.low_speed_timer = 0.0
        self.collision = False
        self.action_list = []
        self.world.tick()

        time.sleep(0.2)
        obs = self.step(None)[0]
        time.sleep(0.2)

        return obs

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
            if self.episode_idx % 2 == 0 and self.num_routes_completed == -1:
                spawn_points_list = [self.world.map.get_spawn_points()[index] for index in next(intersection_routes)]
            else:
                spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)
        else:
            spawn_points_list = [self.world.map.get_spawn_points()[index] for index in next(eval_routes)]
        route_length = 1
        while route_length <= 1:
            self.start_wp, self.end_wp = [self.world.map.get_waypoint(spawn.location) for spawn in
                                          spawn_points_list]
            self.route_waypoints = compute_route_waypoints(self.world.map, self.start_wp, self.end_wp, resolution=1.0)
            route_length = len(self.route_waypoints)
            if route_length <= 1:
                spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)

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
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True
        if self.world is not None:
            actors = self.world.get_actors().filter('vehicle.*')
            for actor in actors:
                actor.destroy()

    def _generate_fdpf_heatmap(self, size=(280, 280)):
        """
        Generate FDPF safety field heatmap as RGB image.
        Returns numpy array of shape (height, width, 3) for pygame display.
        """
        # Create small grid for real-time visualization
        grid_l = np.linspace(-8, 8, 40)  # Lateral
        grid_s = np.linspace(-5, 25, 60)  # Longitudinal
        
        L, S = np.meshgrid(grid_l, grid_s)
        intensity_grid = np.zeros_like(L)
        
        # Get ego speed
        ego_v = self.vehicle.get_velocity()
        ego_speed = np.sqrt(ego_v.x**2 + ego_v.y**2)
        
        # Calculate intensity at each grid point using current FDPF state
        for i in range(len(grid_s)):
            for j in range(len(grid_l)):
                try:
                    intensity, intensity_with_bound, _, _, _, _ = self.fdpf.getIntensityAt(
                        S[i, j], L[i, j], 0.0, ego_speed
                    )
                    intensity_grid[i, j] = min(intensity_with_bound, 3.0)  # Cap for visualization
                except:
                    intensity_grid[i, j] = 0.0
        
        # Create matplotlib figure
        fig = plt.figure(figsize=(3, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot heatmap
        im = ax.contourf(L, S, intensity_grid, levels=15, cmap='RdYlBu_r', alpha=0.9)
        
        # Draw ego vehicle as green rectangle at origin
        from matplotlib.patches import Rectangle
        ego_rect = Rectangle((-1.0, -1.5), 2.0, 3.0, 
                             linewidth=2, edgecolor='black', facecolor='lime', alpha=0.8)
        ax.add_patch(ego_rect)
        
        # Draw lane markings
        ax.axvline(x=-3.5, color='white', linestyle='-', linewidth=2, alpha=0.8)
        ax.axvline(x=3.5, color='white', linestyle='-', linewidth=2, alpha=0.8)
        ax.axvline(x=0, color='yellow', linestyle='--', linewidth=1, alpha=0.6)
        
        # Labels and formatting
        ax.set_xlabel('Lateral (m)', fontsize=8)
        ax.set_ylabel('Longitudinal (m)', fontsize=8)
        ax.set_title(f'FDPF Safety Field\nMax: {intensity_grid.max():.2f}', fontsize=9, fontweight='bold')
        ax.set_xlim(-8, 8)
        ax.set_ylim(-5, 25)
        ax.grid(True, alpha=0.2, linestyle=':')
        ax.tick_params(labelsize=7)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Risk', fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        
        plt.tight_layout()
        
        # Convert to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(canvas.get_width_height()[::-1] + (4,))
        img_rgb = img[:, :, :3]  # Remove alpha channel
        
        plt.close(fig)
        
        # Resize to requested size
        img_resized = cv2.resize(img_rgb, size, interpolation=cv2.INTER_LINEAR)
        
        return img_resized

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

            # Calculate size for right column panels (BEV, RGB, FDPF)
            # Each panel gets 1/3 of window height
            panel_width = self.display.get_size()[1] // 2
            panel_height = self.display.get_size()[1] // 3
            panel_size = (panel_width, panel_height)
            right_x = self.display.get_size()[0] - panel_width
            
            # Top: BEV
            pos_bev = (right_x, 0)
            bev_surface = pygame.surfarray.make_surface(self.bev_spectator_data.swapaxes(0, 1))
            scaled_surface = pygame.transform.scale(bev_surface, panel_size)
            self.display.blit(scaled_surface, pos_bev)

            # Middle: RGB Observation
            pos_obs = (right_x, panel_height)
            obs_surface = pygame.surfarray.make_surface(self.observation_render.swapaxes(0, 1))
            scaled_surface = pygame.transform.scale(obs_surface, panel_size)
            self.display.blit(scaled_surface, pos_obs)

            # Bottom: FDPF Safety Field
            if hasattr(self, 'fdpf') and self.fdpf is not None:
                try:
                    fdpf_image = self._generate_fdpf_heatmap(size=panel_size)
                    pos_fdpf = (right_x, panel_height * 2)
                    fdpf_surface = pygame.surfarray.make_surface(fdpf_image.swapaxes(0, 1))
                    self.display.blit(fdpf_surface, pos_fdpf)
                except Exception as e:
                    pass  # Silent fail if FDPF visualization fails

        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []

        pygame.display.flip()

    def step(self, action):
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")
        if action is not None:
            if self.current_waypoint_index >= len(self.route_waypoints) - 1:
                if not self.eval:
                    self.new_route()
                else:
                    self.success_state = True

            if self.action_space_type == "continuous":
                steer, throttle = [float(a) for a in action]
            elif self.action_space_type == "discrete":
                throttle, steer = discrete_actions[int(action)]

            self.vehicle.control.steer = smooth_action(self.vehicle.control.steer, steer, self.action_smoothing)
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
        self.current_waypoint_index = waypoint_index

        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
                (self.current_waypoint_index + 1) % len(self.route_waypoints)]

        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[
            self.current_waypoint_index % len(self.route_waypoints)]
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(
            self.route_waypoints)

        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        if action is not None:
            self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        self.speed_accum += self.vehicle.get_speed()
        
        # Low speed timeout: terminate if vehicle is stuck (speed < 1 km/h for too long)
        current_speed = self.vehicle.get_speed()
        if current_speed < 3.0:  # Less than 1 km/h
            self.low_speed_timer += 1
            if self.low_speed_timer >= 100:  
                self.terminal_state = True
                print(f"{self.episode_idx}| Terminal:  Vehicle stopped")
        else:
            self.low_speed_timer = 0  # Reset timer when moving

        if self.distance_traveled >= self.max_distance and not self.eval:
            self.success_state = True

        self.distance_from_center_history.append(self.distance_from_center)

        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward

        
        # --- FDPF Update ---
        # 1. Get surrounding vehicles
        vehicles = self.world.get_actors().filter('vehicle.*')
        neighborhood_grids = [[[] for _ in range(3)] for _ in range(3)]
        
        # Ego vars
        ego_trans = self.vehicle.get_transform()
        ego_loc = ego_trans.location
        ego_v = self.vehicle.get_velocity()
        ego_speed = np.sqrt(ego_v.x**2 + ego_v.y**2)
        
        # We define a local Frenet system relative to the route
        # For simplicity in this integration step, we treat:
        # s = distance along route from ego (positive = ahead)
        # l = lateral distance from route center (positive = left)
        
        # Pre-fetch waypoints for projection (window around current)
        window_size = 50
        start_idx = max(0, self.current_waypoint_index - 10)
        end_idx = min(len(self.route_waypoints), self.current_waypoint_index + window_size)
        local_waypoints = [wp[0] for wp in self.route_waypoints[start_idx:end_idx]]
        
        if not local_waypoints:
             local_waypoints = [self.current_waypoint]

        def get_s_l(actor_loc):
            # Find closest waypoint in local window
            min_dist = float('inf')
            closest_wp_idx = -1
            
            for idx, wp in enumerate(local_waypoints):
                d = actor_loc.distance(wp.transform.location)
                if d < min_dist:
                    min_dist = d
                    closest_wp_idx = idx
            
            ref_wp = local_waypoints[closest_wp_idx]
            
            # Simple Frenet approximation
            # s = accumulated distance along waypoints from ego's closest wp
            # But we want relative S to ego.
            # Let's approximate: 
            # Vector from Ego-WP to Actor-WP
            
            # More robust: Project actor onto the vector of the closest waypoint
            wp_trans = ref_wp.transform
            wp_loc = wp_trans.location
            wp_fwd = wp_trans.get_forward_vector()
            wp_right = wp_trans.get_right_vector()
            
            vec = vector(actor_loc - wp_loc)
            
            # Longitudinal projection on WP heading
            s_proj = vec.x * wp_fwd.x + vec.y * wp_fwd.y + vec.z * wp_fwd.z
            
            # Lateral projection
            l_proj = vec.x * wp_right.x + vec.y * wp_right.y + vec.z * wp_right.z
            # l_proj is positive to the right in UE4/Carla (Right Vector), but FDPF might expect Left.
            # FDPF logic: relative_direction = atan2(l, s). 
            # If l is positive left, and we use standard math, it's fine.
            # CARLA Right vector points Right. So Left is -Right.
            l_proj = -l_proj 
            
            # Global S difference
            # S of ego roughly at index 0 of our relative window? No.
            # We calculate index difference
            
            # Distance from ego wp to actor wp along route
            # This is expensive to sum every time.
            # Approx: linear index difference * resolution?
            # Better: just use s_proj + distance_between_wps
            
            # Let's simplify: 
            # s = (Index_Actor - Index_Ego) * resolution + s_proj_correction
            # Assuming ~1m resolution which is typical in carla_route_env if not specified otherwise
            # compute_route_waypoints uses resolution=1.0 default
            
            # Find ego index in local_waypoints
            # We know ego is close to self.current_waypoint_index
            # which roughly corresponds to start_idx + (closest_wp_idx_of_ego)
            
            return closest_wp_idx, s_proj, l_proj

        # Ego relative S, L
        ego_wp_idx, ego_s_proj, ego_l_proj = get_s_l(ego_loc)
        # We treat Ego as S=0, L=ego_l_proj (deviation from center)
        # Actually FDPF expects ego_s, ego_l inputs.
        
        ego_s_fdpf = 0.0
        ego_l_fdpf = ego_l_proj 
        
        # Fill grid
        for veh in vehicles:
            if veh.id == self.vehicle.id:
                continue
                
            veh_loc = veh.get_transform().location
            # Filter distant vehicles (> 50m)
            if veh_loc.distance(ego_loc) > 40:
                continue
                
            w_idx, s_proj, l_proj = get_s_l(veh_loc)
            
            # Calculate relative S
            # S_diff = (Waypoint_Distance) + (Projection_Diff)
            # Waypoint distance approx:
            s_diff_params = (w_idx - ego_wp_idx) * 1.0 # 1.0m resolution
            s_val = ego_s_fdpf + s_diff_params + (s_proj - ego_s_proj)
            l_val = l_proj
            
            # Relative speed
            v_vec = veh.get_velocity()
            v_scalar = np.sqrt(v_vec.x**2 + v_vec.y**2)
            
            # Yaw relative to lane
            # We assume vehicle is roughly aligned with lane for simplicity or calc difference
            # FDPF uses v_x = v * cos(yaw), v_y = v * sin(yaw)
            # Yaw should be relative to the road heading at that point
            veh_yaw = veh.get_transform().rotation.yaw
            wp_yaw = local_waypoints[w_idx].transform.rotation.yaw
            delta_yaw = np.deg2rad(veh_yaw - wp_yaw)
            
            # Determine Grid Cell
            # S: 0->Back, 1->Near, 2->Far
            # L: 0->Right, 1->Center, 2->Left (Relative to Ego Lane?)
            
            # FDPF Grid Usage in calculating intensity:
            # for (s_index, l_index) in [(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]:
            # Processing Near(1) and Far(2) in S.
            
            # Define boundaries for grid
            # Radius approx 10m for "Near", >10m for "Far"?
            # Or based on strictly relative position?
            
            s_rel = s_val - ego_s_fdpf
            l_rel = l_val - ego_l_fdpf
            
            # Oncoming vehicle detection (Town02 dual-lane)
            is_oncoming = (l_rel < -1.75) and (abs(delta_yaw) > np.pi / 2)
            effective_speed = v_scalar * 2.0 if is_oncoming else v_scalar
            
            s_idx = -1
            if -10 < s_rel <= 10: s_idx = 1
            elif s_rel > 10: s_idx = 2
            elif s_rel < -10: s_idx = 0
            
            l_idx = -1
            if l_rel > 1.75: l_idx = 0 # Left
            elif l_rel < -1.75: l_idx = 2 # Right
            else: l_idx = 1 # Center
            
            if s_idx != -1 and l_idx != -1:
                 neighborhood_grids[s_idx][l_idx].append((s_val, l_val, delta_yaw, effective_speed))

        # Update FDPF
        # Dynamically get actual lane width from current waypoint
        if self.current_waypoint is not None:
            actual_lane_width = self.current_waypoint.lane_width
            self.fdpf.lane_width = actual_lane_width if actual_lane_width > 0 else 3.5  # Fallback to 3.5m
        
        self.fdpf.update(neighborhood_grids)
        
        # FDPF inputs for ego
        # Ego delta yaw relative to road
        ego_yaw = ego_trans.rotation.yaw
        ego_wp_yaw = local_waypoints[ego_wp_idx].transform.rotation.yaw
        ego_delta_yaw = np.deg2rad(ego_yaw - ego_wp_yaw)

        (intensity, intensity_with_bound, frenet_direction,
         intensity_s, intensity_l, intensity_bound) = self.fdpf.getIntensityAt(
             ego_s_fdpf, ego_l_fdpf, ego_delta_yaw, ego_speed
         )
         
        self.fdpf_intensity = intensity_with_bound # Use bound version for total risk

        encoded_state = self.encode_state_fn(self)
        self.step_count += 1
        self.total_steps += 1

        if self.activate_render:
            pygame.event.pump()
            if pygame.key.get_pressed()[K_ESCAPE]:
                self.close()
                self.terminal_state = True
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
            'render_array': self.bev_data,
            "centering_factor": centering_factor,
            "angle_factor": angle_factor,
            "distance_std_factor": distance_std_factor,
            "speed": self.vehicle.get_speed(),
            "collision_num": self.collision_num,
            "collision_rate": sum(self.collision_deque) / len(self.collision_deque) if self.collision_deque else 0.0,
            "episode_length": self.step_count,
            "collision_state": self.collision_state,
            "fdpf_intensity": getattr(self, 'fdpf_intensity', 0.0),
        }

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
        while self.observation_buffer is None:
            pass
        obs = self.observation_buffer.copy()
        self.observation_buffer = None
        return obs

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _get_bev_spectator_data(self):
        while self.bev_spectator_data_buffer is None:
            pass
        image = self.bev_spectator_data_buffer.copy()
        self.bev_spectator_data_buffer = None
        return image

    def _get_bev_data(self):
        while self.bev_data_buffer is None:
            pass
        image = self.bev_data_buffer.copy()
        self.bev_data_buffer = None
        return image

    def _on_collision(self, event):
        if get_actor_display_name(event.other_actor) != "Road":
            self.terminal_state = True
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

