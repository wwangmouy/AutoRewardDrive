
import math
import numpy as np


class FDPF():
    """
    Force-based Dynamic Potential Field for autonomous driving safety assessment.
    
    This class computes safety field intensity based on:
    1. Surrounding obstacles (vehicles) with velocity-adaptive distance weighting
    2. Lane boundary constraints with asymmetric penalties
    
    Coordinate System (Frenet):
    - s: longitudinal (along road direction), positive = forward
    - l: lateral (perpendicular to road), positive = left
    """
    
    def __init__(self):
        # Velocity coefficients for distance scaling
        self.Upsilon_coeff_s = 0.3  # m/s - longitudinal
        self.Upsilon_coeff_l = 0.7  # m/s - lateral
        
        # Scaling factors for field shape
        self.major_scale = 2.0   # longitudinal elongation
        self.semi_scale = 0.15  # lateral compression
        
        # Field decay rate
        self.decay_coeff = 0.5
        
        # Lane parameters (set dynamically from CARLA waypoint)
        self.lane_width = 3.5  # default fallback, overwritten by env
        self.margin_distance = 0.8  # safety margin from lane edge
        
        # Boundary penalty weights
        self.centerline_weight = 1.5    # crossing into oncoming traffic
        self.road_shoulder_weight = 2.0  # going off road edge
        
        # Initialize empty grid
        self.neighborhood_grids = [[[] for _ in range(3)] for _ in range(3)]
        
    def update(self, neighborhood_grids):
        """
        Update the neighborhood grid with surrounding vehicle data.
        
        Args:
            neighborhood_grids: 3x3 grid where each cell contains list of 
                               (s, l, delta_yaw, velocity) tuples for vehicles
        """
        if neighborhood_grids is not None:
            self.neighborhood_grids = neighborhood_grids

    def getIntensityAt(self, ego_s, ego_l, ego_delta_yaw, ego_v):
        """
        Calculate safety field intensity at ego vehicle position.
        
        Args:
            ego_s: ego longitudinal position (typically 0 in ego-centric frame)
            ego_l: ego lateral offset from lane center
            ego_delta_yaw: ego heading deviation from road direction (radians)
            ego_v: ego velocity magnitude (m/s)
            
        Returns:
            tuple: (intensity, intensity_with_bound, frenet_direction,
                   intensity_s, intensity_l, intensity_bound)
        """
        intensity_s = 0.0
        intensity_l = 0.0
        
        # Ego velocity components in Frenet frame
        ego_v_s = ego_v * math.cos(ego_delta_yaw)
        ego_v_l = ego_v * math.sin(ego_delta_yaw)
        
        # Process front and near zones: s_index 1 (near), 2 (far)
        # l_index: 0 (left), 1 (center), 2 (right)
        for (s_index, l_index) in [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
            
            neighborhood_grid = self.neighborhood_grids[s_index][l_index]
            
            for obstacle_data in neighborhood_grid:
                # Unpack obstacle data with validation
                if len(obstacle_data) != 4:
                    continue
                s_i, l_i, delta_yaw_i, v_i = obstacle_data
                
                # Skip invalid data
                if v_i < 0 or not math.isfinite(s_i) or not math.isfinite(l_i):
                    continue
                
                # Direction from ego to obstacle
                ds = s_i - ego_s
                dl = l_i - ego_l
                
                # Avoid division by zero for co-located vehicles
                if abs(ds) < 0.01 and abs(dl) < 0.01:
                    continue
                    
                relative_direction = math.atan2(dl, ds)
                
                # Obstacle velocity components
                obstacle_v_s = v_i * math.cos(delta_yaw_i)
                obstacle_v_l = v_i * math.sin(delta_yaw_i)
                
                # Relative closing velocity (only consider approaching obstacles)
                relative_velocity_s = max(ego_v_s - obstacle_v_s, 0.0)
                relative_velocity_l = max(abs(ego_v_l) - abs(obstacle_v_l), 0.0)
                
                # Velocity-adaptive distance coefficients
                coefficient_s = self.Upsilon_coeff_s / (
                    self.major_scale * self.Upsilon_coeff_s + relative_velocity_s + 1e-6)
                coefficient_l = self.Upsilon_coeff_l / (
                    self.semi_scale * self.Upsilon_coeff_l + relative_velocity_l + 1e-6)
                
                # Weighted Euclidean distance
                weighted_dist_sq = (ds ** 2) * coefficient_s + (dl ** 2) * coefficient_l
                euclidean_distance = math.sqrt(max(weighted_dist_sq, 0.0))
                
                # Exponential decay field intensity
                intensity = math.exp(-euclidean_distance * self.decay_coeff)
                
                # Accumulate directional intensity
                intensity_s += intensity * math.cos(relative_direction)
                intensity_l += intensity * math.sin(relative_direction)
        
        # --- Lane Boundary Penalties ---
        # Use dynamic lane_width (set from CARLA waypoint.lane_width)
        effective_lane_width = self.lane_width if self.lane_width > 0 else 3.5
        half_lane = effective_lane_width * 0.5
        
        # Safe zone boundaries (with margin)
        left_boundary = half_lane - self.margin_distance
        right_boundary = -half_lane + self.margin_distance

        # Distance beyond safe zone (positive = violation)
        left_violation = max(0.0, ego_l - left_boundary)   # crossing centerline
        right_violation = max(0.0, right_boundary - ego_l)  # going off road
        
        # Quadratic penalty for boundary violations
        # Both penalties push vehicle back toward center (add to intensity_l)
        intensity_left_bound = self.centerline_weight * (left_violation ** 2)
        intensity_right_bound = self.road_shoulder_weight * (right_violation ** 2)
        
        # Total boundary penalty (always positive)
        intensity_bound = intensity_left_bound + intensity_right_bound
        
        # Combined lateral intensity with boundary effects
        # Left violation pushes right (negative l direction)
        # Right violation pushes left (positive l direction)
        intensity_l_with_bound = intensity_l - intensity_left_bound + intensity_right_bound
                
        # Total intensity magnitude (without bounds)
        intensity = math.sqrt(intensity_s ** 2 + intensity_l ** 2)
        
        # Direction of combined force
        frenet_direction = math.atan2(intensity_l_with_bound, intensity_s) if intensity_s != 0 else 0.0
        
        # Total intensity with boundary effects
        intensity_with_bound = math.sqrt(intensity_s ** 2 + intensity_l_with_bound ** 2) + intensity_bound
        
        return (intensity, intensity_with_bound, frenet_direction,
                intensity_s, intensity_l, intensity_bound)