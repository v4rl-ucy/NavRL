import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import torch
import einops
import numpy as np
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view

from pathlib import Path
import importlib.util

_UTILS_PATH = Path(__file__).resolve().parent / "utils.py"
_spec = importlib.util.spec_from_file_location("navrl_utils", str(_UTILS_PATH))
_navrl_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_navrl_utils)

vec_to_new_frame = _navrl_utils.vec_to_new_frame
vec_to_world = _navrl_utils.vec_to_world
construct_input = _navrl_utils.construct_input

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
import time

class NavigationEnv(IsaacEnv):

    # In one step:
    # 1. _pre_sim_step (apply action) -> step isaac sim
    # 2. _post_sim_step (update lidar)
    # 3. increment progress_buf
    # 4. _compute_state_and_obs (get observation and states, update stats)
    # 5. _compute_reward_and_done (update reward and calculate returns)

    def __init__(self, cfg):
        print("[Navigation Environment]: Initializing Env...")
        # LiDAR params:
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360/self.lidar_hres)

        super().__init__(cfg, cfg.headless)

        # Drone Initialization
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())


        # LiDAR Intialization
        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres,
                vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams)
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams)

        # start and target
        with torch.device(self.device):
            self.target_pos = torch.zeros(self.num_envs, 1, 3)

            # Coordinate change: add target direction variable
            self.target_dir = torch.zeros(self.num_envs, 1, 3)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 1 , 3)

    def _design_scene(self):
        # Initialize a drone in prim /World/envs/envs_0
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] # drone model class
        cfg = drone_model.cfg_cls(force_sensor=False)
        self.drone = drone_model(cfg=cfg)
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.0)])[0]
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        # lighting
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)

        # Ground Plane
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        self.map_range = [20.0, 20.0, 4.5]

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(self.map_range[0]*2, self.map_range[1]*2),
                border_width=5.0,
                num_rows=1,
                num_cols=1,
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.cfg.env.num_obstacles,
                        obstacle_height_mode="range",
                        obstacle_width_range=(0.4, 1.1),
                        obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],
                        obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],
                        platform_width=0.0,
                    ),
                },
            ),
            visual_material = None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=True,
        )
        terrain_importer = TerrainImporter(terrain_cfg)

        if (self.cfg.env_dyn.num_obstacles == 0):
            return
        # Dynamic Obstacles
        # NOTE: we use cuboid to represent 3D dynamic obstacles which can float in the air
        # and the long cylinder to represent 2D dynamic obstacles for which the drone can only pass in 2D
        # The width of the dynamic obstacles is divided into N_w=4 bins
        # [[0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.0]]
        # The height of the dynamic obstacles is divided into N_h=2 bins
        # [[0, 0.5], [0.5, inf]] we want to distinguish 3D obstacles and 2d obstacles
        N_w = 4 # number of width intervals between [0, 1]
        N_h = 2 # number of height: current only support binary
        max_obs_width = 1.0
        self.max_obs_3d_height = 1.0
        self.max_obs_2d_height = 5.0
        self.dyn_obs_width_res = max_obs_width/float(N_w)
        dyn_obs_category_num = N_w * N_h
        self.dyn_obs_num_of_each_category = int(self.cfg.env_dyn.num_obstacles / dyn_obs_category_num)
        self.cfg.env_dyn.num_obstacles = self.dyn_obs_num_of_each_category * dyn_obs_category_num # in case of the roundup error


        # Dynamic obstacle info
        self.dyn_obs_list = []
        self.dyn_obs_state = torch.zeros((self.cfg.env_dyn.num_obstacles, 13), dtype=torch.float, device=self.cfg.device) # 13 is based on the states from sim, we only care the first three which is position
        self.dyn_obs_state[:, 3] = 1. # Quaternion
        self.dyn_obs_goal = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_origin = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_vel = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_step_count = 0 # dynamic obstacle motion step count
        self.dyn_obs_size = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.device) # size of dynamic obstacles


        # helper function to check pos validity for even distribution condition
        def check_pos_validity(prev_pos_list, curr_pos, adjusted_obs_dist):
            for prev_pos in prev_pos_list:
                if (np.linalg.norm(curr_pos - prev_pos) <= adjusted_obs_dist):
                    return False
            return True

        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles) # prefered distance between each dynamic obstacle
        curr_obs_dist = obs_dist
        prev_pos_list = [] # for distance check
        cuboid_category_num = cylinder_category_num = int(dyn_obs_category_num/N_h)
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            # create all origins for 3D dynamic obstacles of this category (size)
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                # random sample an origin until satisfy the evenly distributed condition
                start_time = time.time()
                while (True):
                    ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
                    oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])
                    if (category_idx < cuboid_category_num):
                        oz = np.random.uniform(low=0.0, high=self.map_range[2])
                    else:
                        oz = self.max_obs_2d_height/2. # half of the height
                    curr_pos = np.array([ox, oy])
                    valid = check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist)
                    curr_time = time.time()
                    if (curr_time - start_time > 0.1):
                        curr_obs_dist *= 0.8
                        start_time = time.time()
                    if (valid):
                        prev_pos_list.append(curr_pos)
                        break
                curr_obs_dist = obs_dist
                origin = [ox, oy, oz]
                self.dyn_obs_origin[origin_idx+category_idx*self.dyn_obs_num_of_each_category] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)     
                self.dyn_obs_state[origin_idx+category_idx*self.dyn_obs_num_of_each_category, :3] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)                        
                prim_utils.create_prim(f"/World/Origin{origin_idx+category_idx*self.dyn_obs_num_of_each_category}", "Xform", translation=origin)

            # Spawn various sizes of dynamic obstacles
            if (category_idx < cuboid_category_num):
                # spawn for 3D dynamic obstacles
                obs_width = width = float(category_idx+1) * max_obs_width/float(N_w)
                obs_height = self.max_obs_3d_height
                cuboid_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cuboid",
                    spawn=sim_utils.CuboidCfg(
                        size=[width, width, self.max_obs_3d_height],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cuboid_cfg)
            else:
                radius = float(category_idx-cuboid_category_num+1) * max_obs_width/float(N_w) / 2.
                obs_width = radius * 2
                obs_height = self.max_obs_2d_height
                # spawn for 2D dynamic obstacles
                cylinder_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cylinder",
                    spawn=sim_utils.CylinderCfg(
                        radius = radius,
                        height = self.max_obs_2d_height,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cylinder_cfg)
            self.dyn_obs_list.append(dynamic_obstacle)
            self.dyn_obs_size[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category] \
                = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float, device=self.cfg.device)



    def move_dynamic_obstacle(self):
        # Step 1: Random sample new goals for required update dynamic obstacles
        # Check whether the current dynamic obstacles need new goals
        dyn_obs_goal_dist = torch.sqrt(torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal)**2, dim=1)) if self.dyn_obs_step_count !=0 \
            else torch.zeros(self.dyn_obs_state.size(0), device=self.cfg.device)
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5 # change to a new goal if less than the threshold

        # sample new goals in local range
        num_new_goal = torch.sum(dyn_obs_new_goal_mask)
        sample_x_local = -self.cfg.env_dyn.local_range[0] + 2. * self.cfg.env_dyn.local_range[0] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_y_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[1] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_z_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[2] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_goal_local = torch.cat([sample_x_local, sample_y_local, sample_z_local], dim=1)

        # apply local goal to the global range
        self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local
        # clamp the range if out of the static env range
        self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0])
        self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1])
        self.dyn_obs_goal[:, 2] = torch.clamp(self.dyn_obs_goal[:, 2], min=0., max=self.map_range[2])
        self.dyn_obs_goal[int(self.dyn_obs_goal.size(0)/2):, 2] = self.max_obs_2d_height/2. # for 2d obstacles


        # Step 2: Random sample velocity for roughly every 2 seconds
        if (self.dyn_obs_step_count % int(2.0/self.cfg.sim.dt) == 0):
            self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + (self.cfg.env_dyn.vel_range[1] \
              - self.cfg.env_dyn.vel_range[0]) * torch.rand(self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.cfg.device)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * \
                (self.dyn_obs_goal - self.dyn_obs_state[:, :3])/torch.norm((self.dyn_obs_goal - self.dyn_obs_state[:, :3]), dim=1, keepdim=True)

        # Step 3: Calculate new position update for current timestep
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt


        # Step 4: Update Visualized Location in Simulation
        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category]) 
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        self.dyn_obs_step_count += 1


    def _set_specs(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10

        # Observation Spec
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device),
                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams), device=self.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.device),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)

        # Action Spec
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec, # number of motor
            })
        }).expand(self.num_envs).to(self.device)

        # Reward Spec
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # Done Spec
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)


        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()


    def reset_target(self, env_ids: torch.Tensor):
        if (self.training):
            # decide which side
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)


            # generate random positions
            target_pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            target_pos[:, 0, 2] = heights# height
            target_pos = target_pos * selected_masks + selected_shifts

            # apply target pos
            self.target_pos[env_ids] = target_pos

        else:
            self.target_pos[:, 0, 0] = -5.5
            self.target_pos[:, 0, 1] = 15.
            self.target_pos[:, 0, 2] = 2.


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.reset_target(env_ids)
        if (self.training):
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)

            # generate random positions
            pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            pos[:, 0, 2] = heights# height
            pos = pos * selected_masks + selected_shifts

        else:
            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 0] = -5.5
            pos[:, 0, 1] = 2.
            pos[:, 0, 2] = 2.

        # Coordinate change: after reset, the drone's target direction should be changed
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        # Coordinate change: after reset, the drone's facing direction should face the current goal
        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        diff = self.target_pos[env_ids] - pos
        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        rpy[..., 2] = facing_yaw

        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.prev_drone_vel_w[env_ids] = 0.
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.drone.apply_action(actions)

    def _post_sim_step(self, tensordict: TensorDictBase):
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.move_dynamic_obstacle()
        self.lidar.update(self.dt)

    # get current states/observation
    def _compute_state_and_obs(self):
        gt_root_state = self.drone.get_state(env_frame=False)

        # Keep controller using GT for now
        self.info["drone_state"][:] = gt_root_state[..., :13]

        # Policy observation state starts as GT
        self.root_state = gt_root_state.clone()

        # Replace only NavRL observation x/y and vx/vy with aligned EllipseLIO
        if (
            getattr(self, "_use_lio_for_navrl_obs", False)
            and getattr(self, "_lio_alignment_ready", False)
            and getattr(self, "latest_lio_odom", None) is not None
        ):
            try:
                import numpy as _np

                odom = self.latest_lio_odom
                p = odom.pose.pose.position
                v = odom.twist.twist.linear

                R = self._lio_align_R

                lio_xy = _np.array([p.x, p.y], dtype=float)
                lio0_xy = self._lio0_for_lio_check[:2]
                gt0_xy = self._gt0_for_lio_check[:2]

                lio_rel_xy = lio_xy - lio0_xy
                aligned_xy = R @ lio_rel_xy + gt0_xy

                q = odom.pose.pose.orientation

                q_lio = _np.array([q.x, q.y, q.z, q.w], dtype=float)

                def quat_xyzw_to_R(q):
                    x, y, z, w = q
                    return _np.array([
                        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
                    ], dtype=float)

                R_lio_body = quat_xyzw_to_R(q_lio)

                v_body = _np.array([v.x, v.y, v.z], dtype=float)
                v_lio_odom = R_lio_body @ v_body

                aligned_vel_xy = R @ v_lio_odom[:2]

                self.root_state[0, 0, 0] = float(aligned_xy[0])
                self.root_state[0, 0, 1] = float(aligned_xy[1])

                # Keep GT z for now; LIO z was not reliable enough yet
                self.root_state[0, 0, 7] = float(aligned_vel_xy[0])
                self.root_state[0, 0, 8] = float(aligned_vel_xy[1])

                # Controller also gets LIO x/y and vx/vy
                ctrl_state = gt_root_state[..., :13].clone()
                ctrl_state[0, 0, 0] = float(aligned_xy[0])
                ctrl_state[0, 0, 1] = float(aligned_xy[1])
                ctrl_state[0, 0, 7] = float(aligned_vel_xy[0])
                ctrl_state[0, 0, 8] = float(aligned_vel_xy[1])
                self.info["drone_state"][:] = ctrl_state

            except Exception as e:
                print(f"[LIO-OBS] failed to inject LIO state: {repr(e)}", flush=True)

        # >>>>>>>>>>>>The relevant code starts from here<<<<<<<<<<<<
        # -----------Network Input I: LiDAR range data--------------

        # EllipseLIO collision-sphere observation.
        #
        # Use the most recently completed grid. While a new grid is being
        # generated asynchronously, the policy continues to see the previous
        # complete and internally consistent observation.
        if hasattr(self, "_collision_latest_scan"):
            self.lidar_scan = self._collision_latest_scan.expand(
                self.num_envs,
                -1,
                -1,
                -1,
            )
        else:
            # Startup fallback before the collision client is initialized.
            self.lidar_scan = torch.zeros(
                self.num_envs,
                1,
                self.lidar_hbeams,
                self.lidar_vbeams,
                dtype=torch.float32,
                device=self.device,
            )


        # Visualise the completed EllipseLIO collision observation.
        if (
            hasattr(self, "_collision_latest_scan")
            and hasattr(self, "_collision_ray_dirs")
        ):
            if not hasattr(self, "_collision_draw_started"):
                print("[COLLISION-DRAW] drawing enabled", flush=True)
                self._collision_draw_started = True

            self.debug_draw.clear()

            origin_w = self.lidar.data.pos_w[0].reshape(3)

            dirs_local = self._collision_ray_dirs.to(
                device=self.device,
                dtype=torch.float32,
            )

            # root_state quaternion: [w, x, y, z]
            q = self.root_state[0, 0, 3:7]
            qw, qx, qy, qz = q

            yaw = torch.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )

            c = torch.cos(yaw)
            s = torch.sin(yaw)
            zero = torch.zeros_like(c)
            one = torch.ones_like(c)

            rotation_yaw = torch.stack(
                [
                    torch.stack([c, -s, zero]),
                    torch.stack([s,  c, zero]),
                    torch.stack([zero, zero, one]),
                ]
            )

            dirs_w = torch.matmul(
                dirs_local.reshape(-1, 3),
                rotation_yaw.T,
            ).reshape(
                self.lidar_hbeams,
                self.lidar_vbeams,
                3,
            )

            scan = self._collision_latest_scan[0, 0]

            hit_mask = scan > 0.0

            hit_distances = (
                self._collision_detection_range - scan
            ).clamp(
                min=0.0,
                max=self._collision_detection_range,
            )

            draw_distances = torch.where(
                hit_mask,
                hit_distances,
                torch.full_like(
                    hit_distances,
                    self._collision_detection_range,
                ),
            )

            vectors_w = dirs_w * draw_distances.unsqueeze(-1)

            origins_w = origin_w.view(1, 1, 3).expand_as(vectors_w)

            origins_flat = origins_w.reshape(-1, 3)
            vectors_flat = vectors_w.reshape(-1, 3)
            hits_flat = hit_mask.reshape(-1)

            hit_color = (1.0, 0.0, 0.0, 1.0)
            free_color = (0.0, 1.0, 0.0, 1.0)

            if hits_flat.any():
                self.debug_draw.vector(
                    origins_flat[hits_flat],
                    vectors_flat[hits_flat],
                    color=hit_color,
                )

            if (~hits_flat).any():
                self.debug_draw.vector(
                    origins_flat[~hits_flat],
                    vectors_flat[~hits_flat],
                    color=free_color,
                )

        # ---------Network Input II: Drone's internal states---------
        # a. distance info in horizontal and vertical plane
        rpos = self.target_pos - self.root_state[..., :3]
        distance = rpos.norm(dim=-1, keepdim=True) # start to goal distance
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[..., 2].unsqueeze(-1)

        # b. unit direction vector to goal
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[..., 2] = 0

        rpos_clipped = rpos / distance.clamp(1e-6) # unit vector: start to goal direction
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d) # express in the goal coodinate

        # c. velocity in the goal frame
        vel_w = self.root_state[..., 7:10] # world vel
        vel_g = vec_to_new_frame(vel_w, target_dir_2d)   # coordinate change for velocity

        # final drone's internal states
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1).squeeze(1)

        if (self.cfg.env_dyn.num_obstacles != 0):
            # ---------Network Input III: Dynamic obstacle states--------
            # ------------------------------------------------------------
            # a. Closest N obstacles relative position in the goal frame
            # Find the N closest and within range obstacles for each drone
            dyn_obs_pos_expanded = self.dyn_obs_state[..., :3].unsqueeze(0).repeat(self.num_envs, 1, 1)
            dyn_obs_rpos_expanded = dyn_obs_pos_expanded[..., :3] - self.root_state[..., :3]
            dyn_obs_rpos_expanded[:, int(self.dyn_obs_state.size(0)/2):, 2] = 0.
            dyn_obs_distance_2d = torch.norm(dyn_obs_rpos_expanded[..., :2], dim=2)  # Shape: (1000, 40). calculate 2d distance to each obstacle for all drones
            _, closest_dyn_obs_idx = torch.topk(dyn_obs_distance_2d, self.cfg.algo.feature_extractor.dyn_obs_num, dim=1, largest=False) # pick top N closest obstacle index
            dyn_obs_range_mask = dyn_obs_distance_2d.gather(1, closest_dyn_obs_idx) > self.lidar_range

            # relative distance of obstacles in the goal frame
            closest_dyn_obs_rpos = torch.gather(dyn_obs_rpos_expanded, 1, closest_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            closest_dyn_obs_rpos_g = vec_to_new_frame(closest_dyn_obs_rpos, target_dir_2d)
            closest_dyn_obs_rpos_g[dyn_obs_range_mask] = 0. # exclude out of range obstacles
            closest_dyn_obs_distance = closest_dyn_obs_rpos.norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d = closest_dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_z = closest_dyn_obs_rpos_g[..., 2].unsqueeze(-1)
            closest_dyn_obs_rpos_gn = closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)

            # b. Velocity in the goal frame for the dynamic obstacles
            closest_dyn_obs_vel = self.dyn_obs_vel[closest_dyn_obs_idx]
            closest_dyn_obs_vel[dyn_obs_range_mask] = 0.
            closest_dyn_obs_vel_g = vec_to_new_frame(closest_dyn_obs_vel, target_dir_2d)

            # c. Size of dynamic obstacles in category
            closest_dyn_obs_size = self.dyn_obs_size[closest_dyn_obs_idx] # the acutal size

            closest_dyn_obs_width = closest_dyn_obs_size[..., 0].unsqueeze(-1)
            closest_dyn_obs_width_category = closest_dyn_obs_width / self.dyn_obs_width_res - 1. # convert to category: [0, 1, 2, 3]
            closest_dyn_obs_width_category[dyn_obs_range_mask] = 0.

            closest_dyn_obs_height = closest_dyn_obs_size[..., 2].unsqueeze(-1)
            closest_dyn_obs_height_category = torch.where(closest_dyn_obs_height > self.max_obs_3d_height, torch.tensor(0.0), closest_dyn_obs_height)
            closest_dyn_obs_height_category[dyn_obs_range_mask] = 0.

            # concatenate all for dynamic obstacles
            # dyn_obs_states = torch.cat([closest_dyn_obs_rpos_g, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)
            dyn_obs_states = torch.cat([closest_dyn_obs_rpos_gn, closest_dyn_obs_distance_2d, closest_dyn_obs_distance_z, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)

            # check dynamic obstacle collision for later reward
            closest_dyn_obs_distance_2d_collsion = closest_dyn_obs_rpos[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d_collsion[dyn_obs_range_mask] = float('inf')
            closest_dyn_obs_distance_zn_collision = closest_dyn_obs_rpos[..., 2].unsqueeze(-1).norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_zn_collision[dyn_obs_range_mask] = float('inf')
            dynamic_collision_2d = closest_dyn_obs_distance_2d_collsion <= (closest_dyn_obs_width/2. + 0.3)
            dynamic_collision_z = closest_dyn_obs_distance_zn_collision <= (closest_dyn_obs_height/2. + 0.3)
            dynamic_collision_each = dynamic_collision_2d & dynamic_collision_z
            dynamic_collision = torch.any(dynamic_collision_each, dim=1)

            # distance to dynamic obstacle for reward calculation (not 100% correct in math but should be good enough for approximation)
            closest_dyn_obs_distance_reward = closest_dyn_obs_rpos.norm(dim=-1) - closest_dyn_obs_size[..., 0]/2. # for those 2D obstacle, z distance will not be considered
            closest_dyn_obs_distance_reward[dyn_obs_range_mask] = self.cfg.sensor.lidar_range

        else:
            dyn_obs_states = torch.zeros(self.num_envs, 1, self.cfg.algo.feature_extractor.dyn_obs_num, 10, device=self.cfg.device)
            dynamic_collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.cfg.device)

        # -----------------Network Input Final--------------
        obs = {
            "state": drone_state,
            "lidar": self.lidar_scan,
            "direction": target_dir_2d,
            "dynamic_obstacle": dyn_obs_states
        }


        # -----------------Reward Calculation-----------------
        # a. safety reward for static obstacles
        reward_safety_static = torch.log((self.lidar_range-self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=(2, 3))

        # b. safety reward for dynamic obstacles
        if (self.cfg.env_dyn.num_obstacles != 0):
            reward_safety_dynamic = torch.log((closest_dyn_obs_distance_reward).clamp(min=1e-6, max=self.lidar_range)).mean(dim=-1, keepdim=True)

        # c. velocity reward for goal direction
        vel_direction = rpos / distance.clamp_min(1e-6)
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1)#.clip(max=2.0)

        # d. smoothness reward for action smoothness
        penalty_smooth = (self.drone.vel_w[..., :3] - self.prev_drone_vel_w).norm(dim=-1)

        # e. height penalty reward for flying unnessarily high or low
        penalty_height = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        penalty_height[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)] = ( (self.drone.pos[..., 2] - self.height_range[..., 1] - 0.2)**2 )[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)]
        penalty_height[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)] = ( (self.height_range[..., 0] - 0.2 - self.drone.pos[..., 2])**2 )[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)]

        static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") >  (self.lidar_range - 0.3) # 0.3 collision radius
        collision = static_collision | dynamic_collision

        # Final reward calculation
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.reward = reward_vel + 1. + reward_safety_static * 1.0 + reward_safety_dynamic * 1.0 - penalty_smooth * 0.1 - penalty_height * 8.0
        else:
            self.reward = reward_vel + 1. + reward_safety_static * 1.0 - penalty_smooth * 0.1 - penalty_height * 8.0

        # Terminal reward
        # self.reward[collision] -= 50. # collision

        # Terminate Conditions
        reach_goal = (distance.squeeze(-1) < 0.5)
        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > 4.

        self.terminated = below_bound | above_bound | collision
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) # progress buf is to track the step number

        # update previous velocity for smoothness calculation in the next ieteration
        self.prev_drone_vel_w = self.drone.vel_w[..., :3].clone()

        # # -----------------Training Stats-----------------
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["reach_goal"] = reach_goal.float()
        self.stats["collision"] = collision.float()
        self.stats["truncated"] = self.truncated.float()

        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                },
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
