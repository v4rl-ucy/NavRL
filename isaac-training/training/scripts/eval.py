import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""
eval_photorealistic.py — Photorealistic evaluation of NavRL trained policy

APPROACH: Subclass NavigationEnv, override _design_scene() to load a warehouse
USD and spawn animated people instead of procedural terrain + green primitives.
Everything else (LiDAR, observation pipeline, VelController, PPO) stays identical
to training so the policy sees the same observation format it was trained on.

USAGE (inside Isaac Sim container):
    cd /home/isaac-sim/NavRL/isaac-training/training
    python scripts/eval_photorealistic.py \
        headless=False \
        env.num_envs=1 \
        wandb.mode=disabled \
        +eval_checkpoint=/path/to/checkpoint.pt \
        +warehouse_usd=/isaac-sim/Assets/Isaac/Environments/Simple_Warehouse/full_warehouse.usd \
        +num_people=5

WHAT THIS FILE DOES (section by section):
    1. PhotorealisticEnv — subclass of NavigationEnv that overrides _design_scene()
    2. _design_scene() — loads warehouse USD, spawns drone, spawns people characters
    3. _post_sim_step() — reads animated people positions into dyn_obs_state
       so the observation pipeline sees them as dynamic obstacles
    4. main() — standard eval loop using PPO + VelController (same as eval.py)
"""

import os
import sys
import math
import time
from collections import defaultdict
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

# =============================================================================
# SECTION 1: Isaac Sim Application Setup
# =============================================================================
# Isaac Sim must be initialized BEFORE importing any omni.* modules.
# This is a hard requirement of the Omniverse runtime.

import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from omni.isaac.kit import SimulationApp

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cfg')

@hydra.main(config_path=FILE_PATH, config_name='train', version_base=None)
def main(cfg):
    # =========================================================================
    # Parse extra args from Hydra
    # =========================================================================
    ckpt_path = OmegaConf.select(cfg, 'eval_checkpoint',
        default='/home/isaac-sim/NavRL/quick-demos/ckpts/navrl_checkpoint.pt')
    warehouse_usd = OmegaConf.select(cfg, 'warehouse_usd',
        default='/isaac-sim/Assets/Isaac/Environments/Simple_Warehouse/full_warehouse.usd')
    num_people = int(OmegaConf.select(cfg, 'num_people', default=5))

    # =========================================================================
    # Start Isaac Sim with full rendering (not headless)
    # anti_aliasing: 0=off, 1=FXAA, 2=DLAA, 3=DLSS (best quality)
    # =========================================================================

    sim_app = SimulationApp({
        "headless": cfg.headless,
        "disable_viewport_updates": True,
        "renderer": "MinimalRendering",
        "minimal_shading_mode": 1,  # textured diffuse-ish fast mode
        "anti_aliasing": 0,
        "width": 1280,
        "height": 720,
        "limit_cpu_threads": 24,
        "enable_motion_bvh": True,
    })


    import carb

    carb.settings.get_settings().set_int("/rtx/debugMaterialType", 0)
    settings = carb.settings.get_settings()
    settings.set_int("/rtx/post/dlss/execMode", 0)
    settings.set_bool("/rtx-transient/resourcemanager/texturestreaming/enabled", False)
    settings.set("/rtx/rendermode", "MinimalRendering")
    settings.set_int("/rtx/minimal/mode", 1)
    settings.set_float("/rtx/sceneDb/ambientLightIntensity", 0.5)

    # =========================================================================
    # Enable required extensions
    # These were disabled by default in Isaac Sim 5.1 but are needed by
    # OmniDrones and the animation system
    # =========================================================================
    from omni.isaac.core.utils.extensions import enable_extension
    required_extensions = [
        'omni.isaac.ui',
        'omni.isaac.debug_draw',
        'omni.isaac.cloner',
        'omni.isaac.core_nodes',
        'omni.isaac.sensor',
        'omni.isaac.range_sensor',
        'omni.isaac.dynamic_control',
        'omni.anim.people',          # Animated people extension
        'omni.anim.navigation.core',  # Navigation mesh for people
        'omni.anim.timeline',         # Animation timeline
        'omni.anim.graph.core',       # Animation graph
    ]
    for ext in required_extensions:
        try:
            enable_extension(ext)
            print(f"[EVAL] Enabled extension: {ext}")
        except Exception as e:
            print(f"[EVAL] Warning: Could not enable {ext}: {e}")

    # =========================================================================
    # Now import everything that depends on Isaac Sim being initialized
    # =========================================================================
    import wandb

    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.ros2.bridge")
    enable_extension("isaacsim.sensors.rtx")
    enable_extension("isaacsim.sensors.physics")
    enable_extension("isaacsim.sensors.physx")
    enable_extension("omni.graph.ui_nodes")
    sim_app.update()

    import rclpy
    from sensor_msgs.msg import Imu
    from nav_msgs.msg import Odometry

    print("[EVAL] Enabled ROS2 bridge + RTX sensors before env import", flush=True)

    from env import NavigationEnv
    from ppo import PPO
    from omni_drones.controllers import LeePositionController
    from omni_drones.utils.torchrl.transforms import VelController
    from omni_drones.utils.torchrl import SyncDataCollector
    from torchrl.envs.transforms import TransformedEnv, Compose
    from torchrl.envs.utils import ExplorationType, set_exploration_type
    import omni.isaac.core.utils.prims as prim_utils
    import omni.isaac.orbit.sim as sim_utils
    from omni.isaac.orbit.assets import AssetBaseCfg
    from pxr import Usd, UsdGeom, Sdf
    from pxr import Usd, UsdGeom, UsdPhysics
    # =========================================================================
    # Wandb init (disabled for eval, but PPO code expects it)
    # =========================================================================
    try:
        cd = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        cd = {}
    wandb.init(
        project=cfg.wandb.project,
        name='eval_photorealistic',
        entity=cfg.wandb.entity,
        config=cd,
        mode='disabled',
    )

    # =========================================================================
    # SECTION 2: Create the PhotorealisticEnv
    # =========================================================================
    # We monkey-patch NavigationEnv._design_scene to load our warehouse
    # instead of the procedural terrain. This is cleaner than subclassing
    # because NavigationEnv.__init__ calls _design_scene() internally
    # (via super().__init__), so we need the override in place before __init__.
    #
    # The strategy:
    #   - Save the original _design_scene
    #   - Replace it with our version that loads the warehouse
    #   - After env creation, set up people animation
    # =========================================================================

    # Store warehouse and people config for the patched method
    _eval_config = {
        'warehouse_usd': warehouse_usd,
        'num_people': num_people,
        'people_prims': [],  # Will be populated during scene design
        'asset_root': '/isaac-sim/Assets/Isaac',
    }

    # Save original method
    _original_design_scene = NavigationEnv._design_scene


    def _photorealistic_design_scene(self):
        """
        Replacement for NavigationEnv._design_scene() that loads a warehouse
        USD and spawns people characters instead of procedural terrain.

        WHAT THE ORIGINAL DOES:
        1. Spawns the drone (Hummingbird)
        2. Creates lighting
        3. Creates a ground plane
        4. Creates terrain via TerrainImporter (heightfield with obstacles)
        5. Spawns dynamic obstacles (green cuboids + cylinders)

        WHAT WE DO INSTEAD:
        1. Spawn the drone (SAME — we need the exact same drone)
        2. Load the warehouse USD (replaces ground + terrain + lighting)
        3. Spawn people characters (replaces dynamic obstacles)
        4. Set up the dynamic obstacle tracking arrays (same shape as original)
        """
        from omni_drones.robots.drone import MultirotorBase
        from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg

        # -----------------------------------------------------------------
        # Step 1: Spawn the drone (identical to original)
        # -----------------------------------------------------------------
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name]
        cfg_drone = drone_model.cfg_cls(force_sensor=False)
        self.drone = drone_model(cfg=cfg_drone)
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]
        print("[EVAL] Drone spawned")

        # -----------------------------------------------------------------
        # SLAM IMU sensor: create during scene construction
        # -----------------------------------------------------------------
        try:
            from isaacsim.sensors.physics import IMUSensor, _sensor
            import numpy as np

            self.navrl_imu_path = "/World/envs/env_0/Hummingbird_0/base_link/navrl_imu"

            self.navrl_imu_sensor = IMUSensor(
                prim_path=self.navrl_imu_path,
                name="navrl_imu",
                frequency=125,
                translation=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                linear_acceleration_filter_size=10,
                angular_velocity_filter_size=10,
                orientation_filter_size=10,
            )

            self.navrl_imu_iface = _sensor.acquire_imu_sensor_interface()
            self.navrl_imu_initialized = False

            print("[SLAM-IMU] sensor created during scene construction", flush=True)

        except Exception as e:
            print(f"[SLAM-IMU] scene-time sensor creation failed: {repr(e)}", flush=True)

        # -----------------------------------------------------------------
        # Step 2: Load the warehouse USD
        # -----------------------------------------------------------------
        # The warehouse USD contains floor, walls, shelves, lighting —
        # everything for a photorealistic scene. We add it as a reference
        # under /World/Warehouse.
        #
        # IMPORTANT: The LiDAR raycaster needs to know which prims to cast
        # against. We'll set mesh_prim_paths to include the warehouse.
        # -----------------------------------------------------------------
        warehouse_path = _eval_config['warehouse_usd']
        print(f"[EVAL] Loading warehouse from: {warehouse_path}")

        prim_utils.create_prim(
            "/World/ground/Warehouse",
            usd_path=warehouse_path,
            translation=(0.0, 0.0, 0.0),
        )
        print("[EVAL] Warehouse loaded")

        from pxr import UsdGeom
        stage = prim_utils.get_current_stage()
        wh_prim = stage.GetPrimAtPath("/World/ground/Warehouse")
        if wh_prim:
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            bbox = bbox_cache.ComputeWorldBound(wh_prim)
            rng = bbox.ComputeAlignedRange()
            print(f"[EVAL] Warehouse bounds - Min: {rng.GetMin()}, Max: {rng.GetMax()}")
            # Store bounds for spawn position constraints (with margin for walls)
            #margin = 5.0  # meters from walls
            #self._env_min = [rng.GetMin()[0] + margin, rng.GetMin()[1] + margin]
            #self._env_max = [rng.GetMax()[0] - margin, rng.GetMax()[1] - margin]

            # Hardcoded interior bounds for Simple_Warehouse/full_warehouse.usd
            # TODO: replace with NavMesh-based spawning when baking works

            # full_warehouse.usd
            self._env_min = [-20.0, -5.0]
            self._env_max = [5.0, 25.0]

            '''
            # warehouse_with_multiple_shelfs.usd
            self._env_min = [-8.17, -7.65]
            self._env_max = [6.6, 14.1]
            '''
            print(f"[EVAL] Using manual spawn region: {self._env_min} -> {self._env_max}", flush=True)

            # Safety fallback if the USD bounds are weird/tiny
            if self._env_max[0] <= self._env_min[0] or self._env_max[1] <= self._env_min[1]:
                print("[EVAL] WARNING: invalid derived bounds, falling back to full_warehouse bounds")
                self._env_min = [-20.0, -5.0]
                self._env_max = [5.0, 25.0]

            self._env_z_min = 0.5
            self._env_z_max = min(rng.GetMax()[2] - 1.0, 4.0)  # stay below ceiling
            print(f"[EVAL] Spawn bounds: x=[{self._env_min[0]:.1f}, {self._env_max[0]:.1f}], y=[{self._env_min[1]:.1f}, {self._env_max[1]:.1f}], z=[{self._env_z_min:.1f}, {self._env_z_max:.1f}]")
        # Also add a ground plane (some warehouse USDs don't include one,
        # and we need it for the LiDAR mesh)
        cfg_ground = sim_utils.GroundPlaneCfg(
            color=(0.1, 0.1, 0.1),
            size=(300., 300.)
        )
        cfg_ground.func(
            "/World/defaultGroundPlane",
            cfg_ground,
            translation=(0, 0, -0.01)  # Slightly below warehouse floor
        )

        # Set map_range — defines the navigable area.
        # The warehouse is roughly 20m x 20m. Adjust if needed.
        self.map_range = [20.0, 20.0, 4.5]


        # -----------------------------------------------------------------
        # Step 3: Create terrain mesh for LiDAR
        # -----------------------------------------------------------------
        # The original code uses TerrainImporter which creates a mesh at
        # /World/ground. The LiDAR raycaster is configured to cast against
        # this path. Instead of creating terrain, we create a simple flat
        # ground mesh at /World/ground that the raycaster expects, AND
        # we'll also need to add the warehouse to the raycaster's mesh list.
        #
        # We use TerrainImporter with zero obstacles to create the flat mesh.
        # -----------------------------------------------------------------
        from omni.isaac.orbit.terrains import (
            TerrainImporterCfg, TerrainImporter,
            TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
        )

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
                        num_obstacles=0,  # NO obstacles — warehouse provides them
                        obstacle_height_mode="range",
                        obstacle_width_range=(0.4, 1.1),
                        obstacle_height_range=[1.0, 1.0], #Kon change
                        obstacle_height_probability=[1.0],
                        platform_width=0.0,
                    ),
                },
            ),
            visual_material=None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=False,
        )
        terrain_importer = TerrainImporter(terrain_cfg)
        print("[EVAL] Terrain mesh created for LiDAR")

        # -----------------------------------------------------------------
        # Step 4: Set up dynamic obstacle tracking for people
        # -----------------------------------------------------------------
        # The observation pipeline in _compute_state_and_obs() reads from:
        #   self.dyn_obs_state  — positions of dynamic obstacles
        #   self.dyn_obs_vel    — velocities of dynamic obstacles
        #   self.dyn_obs_size   — sizes of dynamic obstacles
        #   self.dyn_obs_list   — list of RigidObject instances
        #
        # For animated people, we won't use RigidObject. Instead we'll
        # track their positions by reading their prim transforms each step
        # in _post_sim_step(). But we still need to initialize these tensors
        # with the right shapes so the observation pipeline works.
        #
        # We treat each person as a dynamic obstacle with:
        #   - width ≈ 0.6m (human shoulder width)
        #   - height ≈ 1.8m (human height)
        #   - These are "2D obstacles" (tall cylinders) in NavRL's terms
        # -----------------------------------------------------------------
        n_people = _eval_config['num_people']

        # If config says 0 dynamic obstacles but we have people,
        # we need to override the config
        self.cfg.env_dyn.num_obstacles = n_people

        # Initialize tracking tensors (same as original _design_scene)
        self.dyn_obs_list = []  # We won't use RigidObjects for people
        self.dyn_obs_state = torch.zeros(
            (n_people, 13), dtype=torch.float, device=self.cfg.device
        )
        self.dyn_obs_state[:, 3] = 1.  # Quaternion w=1
        self.dyn_obs_goal = torch.zeros(
            (n_people, 3), dtype=torch.float, device=self.cfg.device
        )
        self.dyn_obs_origin = torch.zeros(
            (n_people, 3), dtype=torch.float, device=self.cfg.device
        )
        self.dyn_obs_vel = torch.zeros(
            (n_people, 3), dtype=torch.float, device=self.cfg.device
        )
        self.dyn_obs_step_count = 0
        self.dyn_obs_size = torch.zeros(
            (n_people, 3), dtype=torch.float, device=self.device
        )

        # All people are "2D obstacles" (tall, like cylinders)
        human_width = 0.6   # shoulder width
        human_height = 1.8  # standing height
        self.dyn_obs_size[:, 0] = human_width
        self.dyn_obs_size[:, 1] = human_width
        self.dyn_obs_size[:, 2] = human_height

        # These are needed by the original dynamic obstacle code
        self.max_obs_3d_height = 1.0
        self.max_obs_2d_height = 5.0
        self.dyn_obs_width_res = 0.25  # from original: max_obs_width / N_w
        self.dyn_obs_num_of_each_category = n_people

        # -----------------------------------------------------------------
        # Step 5: Spawn people characters
        # -----------------------------------------------------------------
        # Available characters in the asset pack
        characters = [
            'male_adult_construction_05_new',
            'female_adult_police_02',
            'male_adult_police_04',
            'F_Business_02',
            'M_Medical_01',
            'male_adult_construction_03',
            'female_adult_police_01_new',
            'female_adult_police_03_new',
            'male_adult_construction_01_new',
            'F_Medical_01',
        ]

        asset_root = _eval_config['asset_root']

        # Spawn positions — spread people around the warehouse
        # These will be adjusted once we see the warehouse layout
        '''
        spawn_positions = []
        for i in range(n_people):
            angle = 2 * math.pi * i / n_people
            radius = 5.0 + (i % 3) * 2.0  # Vary radius
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            spawn_positions.append((x, y, 0.0))
        '''
        spawn_positions = []
        for i in range(n_people):
            x = self._env_min[0] + (self._env_max[0] - self._env_min[0]) * np.random.random()
            y = self._env_min[1] + (self._env_max[1] - self._env_min[1]) * np.random.random()
            spawn_positions.append((x, y, 0.0))
        _eval_config['people_prims'] = []

        for i in range(n_people):
            char_name = characters[i % len(characters)]
            char_usd = f"{asset_root}/People/Characters/{char_name}/{char_name}.usd"
            prim_path = f"/World/People/Person_{i}"

            pos = spawn_positions[i]
            prim_utils.create_prim(
                prim_path,
                usd_path=char_usd,
                translation=pos,
            )
            _eval_config['people_prims'].append(prim_path)

            # Initialize obstacle state with spawn position
            self.dyn_obs_state[i, :3] = torch.tensor(
                pos, dtype=torch.float, device=self.cfg.device
            )
            self.dyn_obs_origin[i] = self.dyn_obs_state[i, :3].clone()

            print(f"[EVAL] Spawned person {i}: {char_name} at {pos}")

    # -------------------------------------------------------------------------
    # Patch the move_dynamic_obstacle method
    # -------------------------------------------------------------------------
    # In the original env.py, move_dynamic_obstacle() teleports cuboids/cylinders
    # along straight-line paths. For animated people, we instead read their
    # current positions from the USD stage (they're moved by Omni.Anim.People).
    #
    # However, if Omni.Anim.People isn't set up with commands yet, the people
    # won't move on their own. As a fallback, we use the same teleport logic
    # as the original but update the character prim positions directly.
    # This gives us moving obstacles with correct positions in the observation
    # pipeline, even without walk animations initially.
    # -------------------------------------------------------------------------

    _original_move = NavigationEnv.move_dynamic_obstacle
    _original_post_sim = NavigationEnv._post_sim_step

    def _patched_move_dynamic_obstacle(self):
        """
        Move people using the same random-walk logic as the original,
        but also update the USD prim positions so they visually move.
        """
        # Use the original movement logic (random goals, velocity sampling)
        _original_move(self)

        # Now also update the visual positions of the people prims
        if hasattr(self, '_people_xforms') and self._people_xforms:
            for i, xform in enumerate(self._people_xforms):
                if xform is not None:
                    pos = self.dyn_obs_state[i, :3].cpu().numpy()
                    xform.set_world_pose(
                        position=np.array([pos[0], pos[1], 0.0])
                    )

    # Kon change (added class)
    class StartupGatedPolicy:
        """
        Wrap the trained PPO policy and command zero target velocity until
        the localization/collision-observation startup procedure is complete.

        The wrapped PPO still runs during startup, but its velocity command is
        replaced before VelController converts it into rotor commands.
        """

        def __init__(self, base_policy, env):
            self.base_policy = base_policy
            self.env = env
            self._holding_last_state = None

        def __call__(self, tensordict):
            # PPO writes the world-frame velocity command to:
            # ("agents", "action")
            tensordict = self.base_policy(tensordict)

            navigation_ready = bool(
                getattr(self.env, "navigation_ready", False)
            )

            if not navigation_ready:
                action = tensordict[("agents", "action")]

                # Zero desired world-frame velocity.
                # VelController will turn this into gravity-compensating
                # rotor commands for zero-velocity position hold.
                tensordict.set(
                    ("agents", "action"),
                    torch.zeros_like(action),
                )

            # Print only when the startup state changes.
            if navigation_ready != self._holding_last_state:
                if navigation_ready:
                    print(
                        "[STARTUP-GATE] Navigation ready; "
                        "releasing policy velocity commands",
                        flush=True,
                    )
                else:
                    print(
                        "[STARTUP-GATE] Navigation not ready; "
                        "commanding zero target velocity",
                        flush=True,
                    )

                self._holding_last_state = navigation_ready

            return tensordict


    def quat_to_transform(x, y, z, qx, qy, qz, qw):
            # Quaternion normalization
            norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            if norm != 0:
                qw /= norm
                qx /= norm
                qy /= norm
                qz /= norm

            # Construct rotation matrix from the quaternion
            r11 = 1 - 2*(qy**2 + qz**2)
            r12 = 2*(qx*qy - qw*qz)
            r13 = 2*(qx*qz + qw*qy)

            r21 = 2*(qx*qy + qw*qz)
            r22 = 1 - 2*(qx**2 + qz**2)
            r23 = 2*(qy*qz - qw*qx)

            r31 = 2*(qx*qz - qw*qy)
            r32 = 2*(qy*qz + qw*qx)
            r33 = 1 - 2*(qx**2 + qy**2)

            # Construct the homogeneous transformation
            return  np.array(
            [
                [r11, r12, r13, x],
                [r21, r22, r23, y],
                [r31, r32, r33, z],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=np.float64
            )

    import math
    import numpy as np
    import torch


    def build_angular_mapping(
        source_h_count,
        source_v_count,
        source_h_fov,
        source_v_min,
        source_v_max,
        target_h_count,
        target_v_count,
        target_h_fov,
        target_v_min,
        target_v_max,
    ):
        """
        Build a mapping from each target ray to the four surrounding
        source rays.

        All angular arguments are in radians.

        Returns:
            h_mapping: list of (h_left, h_right)
            v_mapping: list of (v_lower, v_upper)
        """

        # Source horizontal angles.
        # For a full 360-degree scan, endpoint=False avoids duplicating
        # the 0 / 2*pi direction.
        source_h_angles = np.linspace(
            0.0,
            source_h_fov,
            source_h_count,
            endpoint=False,
        )

        # Target horizontal angles.
        target_h_angles = np.linspace(
            0.0,
            target_h_fov,
            target_h_count,
            endpoint=False,
        )

        # Vertical angles include both endpoints.
        source_v_angles = np.linspace(
            source_v_min,
            source_v_max,
            source_v_count,
        )

        target_v_angles = np.linspace(
            target_v_min,
            target_v_max,
            target_v_count,
        )

        # ------------------------------------------------------------
        # Horizontal mapping
        # ------------------------------------------------------------
        h_mapping = []

        for target_angle in target_h_angles:
            # Normalize into source horizontal interval.
            target_angle = target_angle % source_h_fov

            h_right = np.searchsorted(
                source_h_angles,
                target_angle,
                side="left",
            )

            # Horizontal directions wrap around.
            h_right %= source_h_count
            h_left = (h_right - 1) % source_h_count

            # If target exactly equals an existing source ray,
            # use that ray for both sides.
            if np.isclose(
                source_h_angles[h_right],
                target_angle,
                atol=1e-8,
            ):
                h_left = h_right

            h_mapping.append((h_left, h_right))

        # ------------------------------------------------------------
        # Vertical mapping
        # ------------------------------------------------------------
        v_mapping = []

        for target_angle in target_v_angles:
            v_upper = np.searchsorted(
                source_v_angles,
                target_angle,
                side="left",
            )

            # Target below source vertical FOV.
            if v_upper == 0:
                v_lower = 0
                v_upper = 0

            # Target above source vertical FOV.
            elif v_upper >= source_v_count:
                v_lower = source_v_count - 1
                v_upper = source_v_count - 1

            else:
                v_lower = v_upper - 1

                # Exact match.
                if np.isclose(
                    source_v_angles[v_upper],
                    target_angle,
                    atol=1e-8,
                ):
                    v_lower = v_upper

            v_mapping.append((v_lower, v_upper))

        return h_mapping, v_mapping

    def _patched_post_sim_step(self, tensordict):
        """
        Override _post_sim_step to handle people movement and publish IMU.
        """
        _prof_t0 = time.perf_counter()

        if not hasattr(self, "_profile_acc"):
            self._profile_acc = defaultdict(float)
            self._profile_count = 0

        if self.cfg.env_dyn.num_obstacles != 0:
            self.move_dynamic_obstacle()

        _prof_lidar0 = time.perf_counter()
        self.lidar.update(self.dt)
        _prof_lidar1 = time.perf_counter()
        self._profile_acc["policy_lidar_update"] += (_prof_lidar1 - _prof_lidar0)

        # Keep target direction current for NavRL observations
        self.target_dir[:] = self.target_pos - self.root_state[..., :3]

        # Publish simulated IMU at sim-step rate
        _prof_imu0 = time.perf_counter()
        if hasattr(self, "navrl_imu_sensor") and hasattr(self, "navrl_imu_pub"):
            try:
                if not getattr(self, "navrl_imu_initialized", False):
                    self.navrl_imu_sensor.initialize(self.sim.physics_sim_view)
                    self.navrl_imu_sensor.post_reset()
                    self.navrl_imu_sensor.resume()
                    self.navrl_imu_initialized = True

                frame = self.navrl_imu_sensor.get_current_frame(read_gravity=True)

                msg = Imu()

                t = float(frame["time"])

                if hasattr(self, "_last_imu_pub_time") and t <= self._last_imu_pub_time:
                    pass
                else:
                    self._last_imu_pub_time = t

                    msg.header.stamp.sec = int(t)
                    msg.header.stamp.nanosec = int((t - int(t)) * 1e9)
                    msg.header.frame_id = "navrl_imu"

                    # frame["orientation"] is [w, x, y, z]
                    q = frame["orientation"].detach().cpu().numpy()
                    msg.orientation.w = float(q[0])
                    msg.orientation.x = float(q[1])
                    msg.orientation.y = float(q[2])
                    msg.orientation.z = float(q[3])

                    av = frame["ang_vel"].detach().cpu().numpy()
                    msg.angular_velocity.x = float(av[0])
                    msg.angular_velocity.y = float(av[1])
                    msg.angular_velocity.z = float(av[2])

                    la = frame["lin_acc"].detach().cpu().numpy()
                    msg.linear_acceleration.x = float(la[0])
                    msg.linear_acceleration.y = float(la[1])
                    msg.linear_acceleration.z = float(la[2])

                    self.navrl_imu_pub.publish(msg)

            except Exception as e:
                print(f"[SLAM-IMU-ROS] post_sim_step publish failed: {repr(e)}", flush=True)

        _prof_imu1 = time.perf_counter()
        self._profile_acc["imu_block"] += (_prof_imu1 - _prof_imu0)

        _prof_t1 = time.perf_counter()
        self._profile_acc["post_total"] += (_prof_t1 - _prof_t0)
        self._profile_count += 1

        # ===================================================================
        # Listening side
        # ===================================================================

        # Spinning ros node to process pending callbacks
        rclpy.spin_once(
            self.navrl_imu_ros_node,
            timeout_sec=0.0
        )

        # Checking if lio odometry arrived
        if not self.lio_odom_received and self.latest_lio_odom is not None:
            self.lio_odom_received = True

        # -----------------------------
        # Once the lio odometry arrives
        # -----------------------------

        if self.lio_odom_received:
            # ----------------------------------------------------------------
            # Step 1: Calculate world to odometry transform
            # ----------------------------------------------------------------

            # 1.1 Calculate lidar to odometry transform
            x_odom = self.latest_lio_odom.pose.pose.position.x
            y_odom = self.latest_lio_odom.pose.pose.position.y
            z_odom = self.latest_lio_odom.pose.pose.position.z

            qx = self.latest_lio_odom.pose.pose.orientation.x
            qy = self.latest_lio_odom.pose.pose.orientation.y
            qz = self.latest_lio_odom.pose.pose.orientation.z
            qw = self.latest_lio_odom.pose.pose.orientation.w

            t_odom_from_lidar = quat_to_transform(x_odom, y_odom, z_odom, qx, qy, qz, qw)

            # Calculate the target in the odometry frame once
            if self.target_in_odom is None:
                # Assuming that the lidar is aligned with baselink
                # .......... Change an use transforms when publishers are implemented ..........

                # 1.2 Calculate lidar to world transform
                # .......... Again, assuming lidar is aligned with drone ..........
                drone_world_state = self.drone.get_state(env_frame=False)

                drone_world_pose = drone_world_state[0, 0, :13].detach().cpu().numpy()

                position = drone_world_pose[0:3]
                orientation = drone_world_pose[3:7]

                x = position[0]
                y = position[1]
                z = position[2]

                qx = orientation[1]
                qy = orientation[2]
                qz = orientation[3]
                qw = orientation[0]

                t_world_from_lidar = quat_to_transform(x, y, z, qx, qy, qz, qw)


                # 1.3 World to odom calculation
                self.world_to_odom = np.matmul(t_odom_from_lidar, np.linalg.inv(t_world_from_lidar))

                # ------------------------------------------------------------------
                # Step 2: Calculate target in Odom frame

                # Target in Isaac World frame
                target_coords = self.target_pos[0, 0, :].detach().cpu().numpy()

                target_in_world = np.array([target_coords[0], target_coords[1], target_coords[2], 1.0], dtype=np.float64)

                # Target in Odom frame
                self.target_in_odom = np.matmul(self.world_to_odom, target_in_world)

            # Calculate the vector from drone to target
            drone_in_odom = np.array([x_odom, y_odom, z_odom], dtype=np.float64)

            v_drone_to_target = (
                self.target_in_odom[:3]
                - drone_in_odom
            )

            goal_heading = math.atan2(
                v_drone_to_target[1],
                v_drone_to_target[0],
            )

            # Calcualte alignment rotation matrix
            rot_goal = np.array(
            [
                [math.cos(goal_heading), -math.sin(goal_heading), 0.0],
                [math.sin(goal_heading), math.cos(goal_heading), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64
            )

            # Align the grid
            self.aligned_grid = np.matmul(self.collision_check_grid, np.transpose(rot_goal))
            self.aligned_grid += drone_in_odom

            # ---------------------------------------------
            # Step 3. populate service call
            # ---------------------------------------------

            request_attrs = self._collision_request_attrs

            # Increment grid_id
            self.grid_id+=1

            request_attrs["stamp_sec"].set(self.latest_lio_odom.header.stamp.sec)
            request_attrs["stamp_nanosec"].set(self.latest_lio_odom.header.stamp.nanosec)
            request_attrs["frame_id"].set(self.latest_lio_odom.header.frame_id)
            request_attrs["rays_h"].set(self.rays_h)
            request_attrs["rays_v"].set(self.rays_v)
            request_attrs["points"].set(self.aligned_grid.flatten())
            request_attrs["radii"].set(self.radii)
            request_attrs["grid_id"].set(self.grid_id)

            import omni.graph.core as og

            # Only now connect the service client ticking (once)
            if not self.client_activated:

                og.Controller.connect(
                    self.gate_exec_out_attr,
                    self.client_exec_in_attr,
                )

                print(
                    "[COLLISION-SERVICE-CLIENT] "
                    "Physics-step gate connected; request interval=2 steps",
                    flush=True,
                )

                self.client_activated = True

            self._last_collision_response = {
                "collision_grid": og.Controller.get(self._collision_response_grid_attr),
                "success": og.Controller.get(self._collision_success_attr),
                "message": og.Controller.get(self._collision_message_attr),
                "grid_id": og.Controller.get(self._collision_grid_id_attr)
            }

            self._collision_response_count +=1
            self._collision_response_window_start_sim = None

            response = self._last_collision_response

            # ---------------------------------------------------------------
            # Populate the NavRL lidar observation from a NEW completed
            # EllipseLIO batch response.
            # ---------------------------------------------------------------

            response_grid_id = int(response["grid_id"])

            expected_grid_size = (
                self.rays_h
                * self.rays_v
                * len(self.collision_sample_distances)
            )

            # Process only:
            #   1. a real response (grid_id > 0),
            #   2. a newer response than the one already consumed,
            #   3. a response that the server says was completed successfully,
            #   4. a response with the expected number of collision values.
            if (
                response_grid_id > self._last_processed_collision_grid_id
                and bool(response["success"])
                and len(response["collision_grid"]) == expected_grid_size
            ):

                # -----------------------------------------------------------
                # 1. Reconstruct raw collision tensor
                #
                # Shape:
                #   [collision horizontal rays,
                #    collision vertical rays,
                #    radial samples]
                # -----------------------------------------------------------
                collision_grid = np.asarray(
                    response["collision_grid"],
                    dtype=np.bool_,
                ).reshape(
                    self.rays_h,
                    self.rays_v,
                    len(self.collision_sample_distances),
                )

                # Keep the completed raw grid too, since it is useful for
                # debugging/visualisation later.
                self.latest_complete_collision_grid[...] = collision_grid

                # -----------------------------------------------------------
                # 2. Collapse radial dimension.
                #
                # For every new collision ray, find the closest occupied
                # sphere centre.
                #
                # No collision -> collision_detection_range.
                # -----------------------------------------------------------
                new_distance_scan = np.full(
                    (self.rays_h, self.rays_v),
                    self.collision_detection_range,
                    dtype=np.float32,
                )

                # collision_sample_distances is stored FAR -> NEAR,
                # so iterate backwards to encounter the nearest sphere first.
                for h in range(self.rays_h):
                    for v in range(self.rays_v):
                        for k in range(
                            len(self.collision_sample_distances) - 1,
                            -1,
                            -1,
                        ):
                            if collision_grid[h, v, k]:
                                new_distance_scan[h, v] = float(
                                    self.collision_sample_distances[k]
                                )
                                break

                # -----------------------------------------------------------
                # 3. Map the new angular discretisation to the angular
                #    discretisation expected by the pretrained NavRL policy.
                #
                # Each old/policy ray uses the closest obstacle among the
                # four surrounding new collision rays.
                # -----------------------------------------------------------
                policy_distance_scan = np.full(
                    (
                        self.lidar_hbeams,
                        self.lidar_vbeams,
                    ),
                    self.lidar_range,
                    dtype=np.float32,
                )

                for h_policy, (h_left, h_right) in enumerate(
                    self._collision_h_mapping
                ):
                    for v_policy, (v_lower, v_upper) in enumerate(
                        self._collision_v_mapping
                    ):

                        d0 = new_distance_scan[h_left,  v_lower]
                        d1 = new_distance_scan[h_left,  v_upper]
                        d2 = new_distance_scan[h_right, v_lower]
                        d3 = new_distance_scan[h_right, v_upper]

                        # Conservative angular mapping:
                        # policy ray sees the closest obstacle among its
                        # surrounding higher-resolution collision rays.
                        policy_distance_scan[h_policy, v_policy] = min(
                            d0,
                            d1,
                            d2,
                            d3,
                        )

                # -----------------------------------------------------------
                # 4. Restrict distances to what the OLD NavRL sensor could
                #    represent.
                #
                # Collision model range: 4.0347 m
                # Trained NavRL range:    4.0 m (current config)
                #
                # Anything outside the old range should look like max range.
                # -----------------------------------------------------------
                policy_distance_scan = np.clip(
                    policy_distance_scan,
                    0.0,
                    self.lidar_range,
                )

                # -----------------------------------------------------------
                # 5. Reproduce the representation used during NavRL training:
                #
                #       observation = lidar_range - distance
                #
                # No obstacle at max range -> 0
                # Close obstacle           -> large value
                # -----------------------------------------------------------
                policy_scan = (
                    self.lidar_range
                    - policy_distance_scan
                )

                # -----------------------------------------------------------
                # 6. Convert:
                #
                # [H_policy, V_policy]
                #         ->
                # [1, 1, H_policy, V_policy]
                #
                # matching _collision_latest_scan / NavRL lidar observation.
                # -----------------------------------------------------------
                policy_scan_tensor = torch.as_tensor(
                    policy_scan,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0).unsqueeze(0)

                self._collision_latest_scan.copy_(
                    policy_scan_tensor
                )

                # Mark this response as consumed.
                self._last_processed_collision_grid_id = response_grid_id

                # The first complete collision observation means localization
                # and static-obstacle sensing are now both available.
                if not self.navigation_ready:
                    self.navigation_ready = True

                    print(
                        "[COLLISION-OBS] First valid mapped observation received; "
                        "navigation ready",
                        flush=True,
                    )

    # =========================================================================
    # Apply patches
    # =========================================================================
    NavigationEnv._design_scene = _photorealistic_design_scene
    NavigationEnv.move_dynamic_obstacle = _patched_move_dynamic_obstacle
    NavigationEnv._post_sim_step = _patched_post_sim_step

    _original_reset_target = NavigationEnv.reset_target
    _original_reset_idx = NavigationEnv._reset_idx

    def _patched_reset_target(self, env_ids):
        if hasattr(self, "_env_min"):
            xmin, ymin = self._env_min
            xmax, ymax = self._env_max

            x = xmin + (xmax - xmin) * torch.rand(env_ids.size(0), device=self.device)
            y = ymin + (ymax - ymin) * torch.rand(env_ids.size(0), device=self.device)
            z = self._env_z_min + (self._env_z_max - self._env_z_min) * torch.rand(
                env_ids.size(0), device=self.device
            )

            self.target_pos[env_ids, 0, 0] = x
            self.target_pos[env_ids, 0, 1] = y
            self.target_pos[env_ids, 0, 2] = z
        else:
            _original_reset_target(self, env_ids)


    def _patched_reset_idx(self, env_ids):
        self.drone._reset_idx(env_ids, self.training)
        self.reset_target(env_ids)

        if hasattr(self, "_env_min"):
            xmin, ymin = self._env_min
            xmax, ymax = self._env_max

            x = xmin + (xmax - xmin) * torch.rand(len(env_ids), device=self.device)
            y = ymin + (ymax - ymin) * torch.rand(len(env_ids), device=self.device)
            z = self._env_z_min + (self._env_z_max - self._env_z_min) * torch.rand(
                len(env_ids), device=self.device
            )

            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 0] = x
            pos[:, 0, 1] = y
            pos[:, 0, 2] = z
        else:
            return _original_reset_idx(self, env_ids)

        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        diff = self.target_pos[env_ids] - pos
        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        rpy[..., 2] = facing_yaw

        from omni_drones.utils.torch import euler_to_quaternion
        rot = euler_to_quaternion(rpy)

        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.prev_drone_vel_w[env_ids] = 0.
        self.height_range[env_ids, 0, 0] = torch.min(
            pos[:, 0, 2], self.target_pos[env_ids, 0, 2]
        )
        self.height_range[env_ids, 0, 1] = torch.max(
            pos[:, 0, 2], self.target_pos[env_ids, 0, 2]
        )
        self.stats[env_ids] = 0.
    NavigationEnv.reset_target = _patched_reset_target
    NavigationEnv._reset_idx = _patched_reset_idx


    # =========================================================================
    # SECTION 3: Create the environment
    # =========================================================================
    print("[EVAL] Creating environment...")
    env = NavigationEnv(cfg)

    # Visual-only target marker
    try:
        import omni.isaac.core.utils.prims as prim_utils
        import omni.isaac.orbit.sim as sim_utils

        target_marker_cfg = sim_utils.SphereCfg(
            radius=0.25,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                emissive_color=(1.0, 0.0, 0.0),
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )

        target_marker_cfg.func(
            "/World/DebugTargetMarker",
            target_marker_cfg,
            translation=(0.0, 0.0, 0.0),
        )

        env._target_marker_path = "/World/DebugTargetMarker"
        print("[DEBUG-TARGET] visual marker created", flush=True)

    except Exception as e:
        print(f"[DEBUG-TARGET] failed to create marker: {repr(e)}", flush=True)

    # Render RTX sensors every N env/sim steps.
    # With dt ~= 0.008 and sim rate ~=125 Hz:
    # N=12 gives about 10.4 Hz target render/LiDAR rate.
    env._render_counter = 0

    def render_every_12_steps(substep):
        if substep != 0:
            return False

        env._render_counter += 1

        # physics is 125 Hz, so 12 or 13 gives about 10 Hz rendering/lidar trigger
        return env._render_counter % 6 == 0

    env.enable_render(render_every_12_steps)

    try:
        print("[SIM] physics_dt:", env.sim.get_physics_dt(), flush=True)
        print("[SIM] physics_hz:", 1.0 / env.sim.get_physics_dt(), flush=True)
        print("[SIM] rendering_dt:", env.sim.get_rendering_dt(), flush=True)
        print("[SIM] rendering_hz:", 1.0 / env.sim.get_rendering_dt(), flush=True)
    except Exception as e:
        print(f"[SIM] could not read sim dt: {repr(e)}", flush=True)

    # -----------------------------------------------------------------
    # ROS2 Clocks
    # -----------------------------------------------------------------

    # ROS2 Clock Publisher
    try:
        import omni.graph.core as og
        import omni.usd
        from isaacsim.core.utils.stage import get_current_stage
        import omni.timeline
        from omni.graph.core import GraphPipelineStage
        #enable_extension("omni.graph.action")

        og.Controller.edit(
            {"graph_path": "/ActionGraph_Clock", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ],
            },
        )

        omni.timeline.get_timeline_interface().play()
        print("[ROS-TEST] Created /clock publisher graph", flush=True)
    except Exception as e:
        print(f"[ROS-TEST] Failed to create /clock publisher: {repr(e)}", flush=True)

    # -----------------------------------------------------------------
    # SLAM PhysX SDK LiDAR ROS2 publisher
    # -----------------------------------------------------------------
    ENABLE_SLAM_PHYSX_LIDAR = True

    if ENABLE_SLAM_PHYSX_LIDAR:
        print("[SLAM-LIDAR] setting up PhysX SDK LiDAR", flush=True)

        try:
            import omni
            from isaacsim.sensors.physx import _range_sensor
            from pxr import Gf
            from sensor_msgs.msg import PointCloud2, PointField

            parent = "/World/envs/env_0/Hummingbird_0/base_link"

            # PhysX Lidar uses PhysX raycasts.
            # rotation_rate=0.0 means all rays are fired every sensor update

            _, lidar_prim = omni.kit.commands.execute(
                "RangeSensorCreateLidar",
                path="slam_physx_lidar",
                parent="/World/envs/env_0/Hummingbird_0/base_link",
                min_range=0.2,
                max_range=30.0,
                draw_points=False,
                draw_lines=False,
                horizontal_fov=360.0,
                vertical_fov=45.0,
                horizontal_resolution=360.0 / 1024.0,
                vertical_resolution=45.0 / 128.0,
                rotation_rate=0.0,
                high_lod=True,
                yaw_offset=0.0,
                enable_semantics=False,
            )

            env.physx_lidar_path = str(lidar_prim.GetPath())
            print("[PHYSX-LIDAR] legacy lidar path:", env.physx_lidar_path, flush=True)

            print("[PHYSX-LIDAR] legacy lidar path:", env.physx_lidar_path, flush=True)

            import omni.graph.core as og

            ENABLE_NATIVE_LIDAR_ROS = True
            if ENABLE_NATIVE_LIDAR_ROS:

                og.Controller.edit(
                    {"graph_path": "/ActionGraph_PhysX_Lidar_ROS", "evaluator_name": "execution"},
                    {
                        og.Controller.Keys.CREATE_NODES: [
                            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                            ("ReadLidarPointCloud", "isaacsim.sensors.physx.IsaacReadLidarPointCloud"),
                            ("PublishPointCloud", "isaacsim.ros2.bridge.ROS2PublishPointCloud"),
                        ],
                        og.Controller.Keys.CONNECT: [
                            ("OnPlaybackTick.outputs:tick", "ReadLidarPointCloud.inputs:execIn"),
                            ("ReadLidarPointCloud.outputs:execOut", "PublishPointCloud.inputs:execIn"),
                            ("ReadLidarPointCloud.outputs:data", "PublishPointCloud.inputs:data"),
                            ("ReadSimTime.outputs:simulationTime", "PublishPointCloud.inputs:timeStamp"),
                        ],
                        og.Controller.Keys.SET_VALUES: [
                            ("ReadLidarPointCloud.inputs:lidarPrim", env.physx_lidar_path),
                            ("PublishPointCloud.inputs:topicName", "navrl_lidar_points"),
                            ("PublishPointCloud.inputs:frameId", "slam_physx_lidar"),
                        ],
                    },
                )

            else:

                print("[PHYSX-LIDAR] Native ROS2 PointCloud graph DISABLED", flush=True)

            try:
                stage = omni.usd.get_context().get_stage()
                lidar_usd_prim = stage.GetPrimAtPath(env.physx_lidar_path)

                print("[PHYSX-LIDAR] USD prim attributes:", flush=True)
                print(f"  path = {env.physx_lidar_path}", flush=True)
                print(f"  valid = {lidar_usd_prim.IsValid()}", flush=True)
                print(f"  type = {lidar_usd_prim.GetTypeName()}", flush=True)

                for attr in lidar_usd_prim.GetAttributes():
                    name = attr.GetName()
                    if (
                        "fov" in name.lower()
                        or "resolution" in name.lower()
                        or "vertical" in name.lower()
                        or "horizontal" in name.lower()
                        or "range" in name.lower()
                        or "rotation" in name.lower()
                    ):
                        try:
                            print(f"  {name} = {attr.Get()}", flush=True)
                        except Exception as e:
                            print(f"  {name} = <failed: {e}>", flush=True)
            except Exception as e:
                print(f"[PHYSX-LIDAR] attribute inspection failed: {repr(e)}", flush=True)

            if not rclpy.ok():
                rclpy.init(args=None)

            env.navrl_imu_ros_node = rclpy.create_node("navrl_imu_publisher")
            print("[SLAM-LIDAR] Native OmniGraph publishing PhysX /navrl_lidar_points", flush=True)
        except Exception as e:
            print(f"[SLAM-LIDAR] PhysX setup failed: {repr(e)}", flush=True)

    else:
        print("[SLAM-LIDAR] FULLY DISABLED", flush=True)

    # -----------------------------------------------------------------
    # SLAM IMU ROS2 Python publisher
    # -----------------------------------------------------------------
    try:
        if not rclpy.ok():
            rclpy.init(args=None)

        if not hasattr(env, "navrl_imu_ros_node"):
            env.navrl_imu_ros_node = rclpy.create_node("navrl_imu_publisher")
        env.navrl_imu_pub = env.navrl_imu_ros_node.create_publisher(
            Imu,
            "/navrl_imu",
            10,
        )

        print("[SLAM-IMU-ROS] publisher created on /navrl_imu", flush=True)
    except Exception as e:
        print(f"[SLAM-IMU-ROS] publisher creation failed: {repr(e)}", flush=True)

    # =========================================================================
    # EllipseLio Odometry Subscriber
    # =========================================================================
    env.latest_lio_odom = None

    def _lio_odom_cb(msg):
        env.latest_lio_odom = msg

    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

    lio_odom_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )

    env.lio_odom_sub = env.navrl_imu_ros_node.create_subscription(
        Odometry,
        "/ellipselio/odom",
        _lio_odom_cb,
        lio_odom_qos,
    )
    print("[LIO-CHECK] subscribed to /ellipselio_odom", flush=True)

    # ========================================================================
    # EllipseLio Batch Collision Checking Client
    # ========================================================================
    try:
        import omni.graph.core as og

        enable_extension("omni.graph.action")
        sim_app.update()

        graph_path = "/ActionGraph_BatchCollisionServiceClient"
        client_path = f"{graph_path}/ROS2ServiceClient"

        # --------------------------------------------------------------------
        # 1. Create and configure the client without a trigger.
        #
        # The first app update allows the generic client to inspect
        # ellipselio/srv/BatchCheckCollision and dynamically create its
        # Request/Response attributes.
        # --------------------------------------------------------------------

        # Step 1: create the client and its gated physics trigger.
        from omni.graph.core import GraphPipelineStage

        og.Controller.edit(
            {
                "graph_path": graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    (
                        "OnPhysicsStep",
                        "isaacsim.core.nodes.OnPhysicsStep",
                    ),
                    (
                        "CollisionGate",
                        "isaacsim.core.nodes.IsaacSimulationGate",
                    ),
                    (
                        "ROS2Context",
                        "isaacsim.ros2.bridge.ROS2Context",
                    ),
                    (
                        "ROS2ServiceClient",
                        "isaacsim.ros2.bridge.OgnROS2ServiceClient",
                    ),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        "OnPhysicsStep.outputs:step",
                        "CollisionGate.inputs:execIn",
                    ),
                    (
                        "ROS2Context.outputs:context",
                        "ROS2ServiceClient.inputs:context",
                    ),
                ],
                og.Controller.Keys.SET_VALUES: [
                    (
                        "CollisionGate.inputs:step",
                        12,
                    ),
                ],
            },
        )

        sim_app.update()

        service_values = {
            "inputs:messagePackage": "ellipselio",
            "inputs:messageSubfolder": "srv",
            "inputs:messageName": "BatchCheckCollision",
            "inputs:serviceName": "/ellipselio/batch_check_collision",
        }

        for attribute_name, value in service_values.items():
            attribute = og.Controller.attribute(
                f"{client_path}.{attribute_name}"
            )

            if not attribute.is_valid():
                raise RuntimeError(
                    f"Missing base service-client attribute: {attribute_name}"
                )

            og.Controller.set(attribute, value)

        print(
            "[COLLISION-SERVICE-CLIENT] Configured service type:",
            flush=True,
        )

        for attribute_name in service_values:
            attribute = og.Controller.attribute(
                f"{client_path}.{attribute_name}"
            )
            print(
                f"  {attribute_name}="
                f"{og.Controller.get(attribute)!r}",
                flush=True,
            )

        # Wait until the generic node expands the custom service fields.
        probe_path = (
            f"{client_path}.inputs:Request:header:stamp:sec"
        )

        for update_index in range(50):
            sim_app.update()

            probe_attribute = og.Controller.attribute(probe_path)

            if probe_attribute.is_valid():
                print(
                    "[COLLISION-SERVICE-CLIENT] "
                    f"Interface expanded after {update_index + 1} updates",
                    flush=True,
                )
                break
        else:
            client_node = og.Controller.node(client_path)

            available_attributes = []
            if client_node is not None:
                available_attributes = [
                    attribute.get_name()
                    for attribute in client_node.get_attributes()
                ]

            raise RuntimeError(
                "CheckCollision Request/Response attributes were not generated. "
                f"Available attributes: {available_attributes}"
            )
        # --------------------------------------------------------------------
        # Step 2. Verify that the custom service fields were created.
        # --------------------------------------------------------------------
        required_attributes = [
            "inputs:Request:header:stamp:sec",
            "inputs:Request:header:stamp:nanosec",
            "inputs:Request:header:frame_id",
            "inputs:Request:rays_h",
            "inputs:Request:rays_v",
            "inputs:Request:points",
            "inputs:Request:radii",
            "inputs:Request:grid_id",
            "outputs:Response:collision_grid",
            "outputs:Response:success",
            "outputs:Response:message",
            "outputs:Response:grid_id",
        ]

        for attribute_name in required_attributes:
            attribute = og.Controller.attribute(
                f"{client_path}.{attribute_name}"
            )
            if not attribute.is_valid():
                raise RuntimeError(
                    "Collision service client is missing attribute: "
                    f"{attribute_name}"
                )

        print(
            "[COLLISION-SERVICE-CLIENT] "
            "CheckCollision interface loaded successfully",
            flush=True,
        )

        # --------------------------------------------------------------------
        # 2. Physics-motivated collision grid observation model
        # --------------------------------------------------------------------

        # ---------- Step 1 ----------
        # Define grid dimensions.
        # These should be replaced by the configuration file vlaues.

        env.collision_detection_range = 4.0347

        env.rays_h = 37
        env.rays_v = 16

        env.fov_h = 2.0*math.pi
        env.fov_v = math.radians(104.0)

        env.separation_h = env.fov_h/env.rays_h
        env.separation_v = env.fov_v/(env.rays_v -1)

        # Sample distances along each ray.
        env.collision_sample_distances = np.array([
            4.0347, 3.7065, 3.4049, 3.1279, 2.8734, 2.6397,
            2.4249, 2.2276, 2.0464, 1.8799, 1.7270, 1.5865,
            1.4574, 1.3388, 1.2299, 1.1298, 1.0379, 0.9535,
            0.8759, 0.8046, 0.7392, 0.6790, 0.6238, 0.5731,
            0.5264, 0.4836, 0.4443, 0.4081, 0.3749, 0.3444,
            0.3164, 0.2907
        ],
        dtype=np.float64
        )

        # Checking radii at each distance
        env.radii = np.array([
            0.3422, 0.3143, 0.2888, 0.2653, 0.2437, 0.2239,
            0.2056, 0.1889, 0.1735, 0.1594, 0.1465, 0.1345,
            0.1236, 0.1135, 0.1043, 0.0958, 0.0880, 0.0809,
            0.0743, 0.0682, 0.0627, 0.0576, 0.0529, 0.0486,
            0.0446, 0.0410, 0.0377, 0.0346, 0.0318, 0.0292,
            0.0268, 0.0246
        ],
        dtype=np.float64,
        )

        # ---------- Step 2 ----------
        # Compute collison checking points
        env.collision_check_grid = np.empty(
                (
                    env.rays_h,
                    env.rays_v,
                    len(env.collision_sample_distances),
                    3
                ),
                dtype=np.float64
            )

        # Iterate over each ray
        for i in range(env.rays_h):
           for j in range(env.rays_v):
               # calculate direction unit vector
               angle_h = env.separation_h*i
               angle_v = env.separation_v*j - env.fov_v/2 # Centering vertical FOV to 0

               u = np.array([
                   math.cos(angle_v)*math.cos(angle_h),
                   math.cos(angle_v)*math.sin(angle_h),
                   math.sin(angle_v)
               ])

               # calculate point centers along the ray
               for k in range(len(env.collision_sample_distances)):
                   env.collision_check_grid[i][j][k] = env.collision_sample_distances[k]*u

        # ---------- Step 3 ----------
        # Initialise Startup State.
        # To align the grid to the target, we need to get
        # a pose estimate from ellipselio. Until then,
        # the state is set to uninitialised.

        env.lio_odom_received = False
        env.navigation_ready = False
        env.client_activated = False
        env.grid_id = 0

        # Initializing placeholders for alignment
        env.world_to_odom = None
        env.target_in_odom = None

        env.goal_aligned_collision_check_grid = np.zeros_like(
                env.collision_check_grid,
                dtype =np.float64
        )

        env.latest_complete_collision_grid = np.zeros(
                (
                    env.rays_h,
                    env.rays_v,
                    len(env.collision_sample_distances),
                ),
                dtype=np.bool_
            )

        env._collision_latest_scan = torch.zeros(
            (
                1,
                1,
                env.lidar_hbeams,
                env.lidar_vbeams,
            ),
            dtype=torch.float32,
            device=env.device,
        )

        # Compute the mapping between the observation space
        # and the trained on observation space.
        env._collision_h_mapping, env._collision_v_mapping = (
            build_angular_mapping(
                # New collision-sphere grid
                source_h_count=env.rays_h,
                source_v_count=env.rays_v,
                source_h_fov=env.fov_h,
                source_v_min=-env.fov_v / 2.0,
                source_v_max= env.fov_v / 2.0,

                # Original NavRL policy grid
                target_h_count=env.lidar_hbeams,
                target_v_count=env.lidar_vbeams,
                target_h_fov=2.0 * math.pi,
                target_v_min=math.radians(env.lidar_vfov[0]),
                target_v_max=math.radians(env.lidar_vfov[1]),
            )
        )

        # --------------------------------------------------------------------
        # 3. Only now create OnPlaybackTick and connect it to execIn.
        #
        # Once playback begins, each playback tick sends the configured
        # request. This is only for communication testing.
        # --------------------------------------------------------------------
        env.gate_exec_out_attr = og.Controller.attribute(
            f"{graph_path}/CollisionGate.outputs:execOut"
        )

        env.client_exec_in_attr = og.Controller.attribute(
            f"{client_path}.inputs:execIn"
        )

        if not env.gate_exec_out_attr.is_valid():
            raise RuntimeError(
                "CollisionGate execOut attribute is invalid"
            )

        if not env.client_exec_in_attr.is_valid():
            raise RuntimeError(
                "Service client execIn attribute is invalid"
            )

        print(
            "[COLLISION-SERVICE-CLIENT] "
            "Playback-tick gate connected; request interval=2 ticks",
            flush=True,
        )

        # Keep the paths/attributes on env so _post_sim_step can inspect them.
        env._collision_client_path = client_path

        env._collision_request_attrs = {
            "stamp_sec": og.Controller.attribute(
                f"{client_path}.inputs:Request:header:stamp:sec"
            ),
            "stamp_nanosec": og.Controller.attribute(
                f"{client_path}.inputs:Request:header:stamp:nanosec"
            ),
            "frame_id": og.Controller.attribute(
                f"{client_path}.inputs:Request:header:frame_id"
            ),
            "rays_h": og.Controller.attribute(
                f"{client_path}.inputs:Request:rays_h"
            ),
            "rays_v": og.Controller.attribute(
                f"{client_path}.inputs:Request:rays_v"
            ),
            "points": og.Controller.attribute(
                f"{client_path}.inputs:Request:points"
            ),
            "radii": og.Controller.attribute(
                f"{client_path}.inputs:Request:radii"
            ),
            "grid_id": og.Controller.attribute(
                 f"{client_path}.inputs:Request:grid_id"
             )
        }

        env._collision_response_grid_attr = og.Controller.attribute(
            f"{client_path}.outputs:Response:collision_grid"
        )
        env._collision_success_attr = og.Controller.attribute(
            f"{client_path}.outputs:Response:success"
        )
        env._collision_message_attr = og.Controller.attribute(
            f"{client_path}.outputs:Response:message"
        )
        env._collision_grid_id_attr = og.Controller.attribute(
            f"{client_path}.outputs:Response:grid_id"
        )

        env._last_collision_response = None

        env._collision_response_count = 0
        env._collision_response_window_start_sim = None
        env._last_processed_collision_grid_id = 0

    except Exception as e:
        print(
            f"[COLLISION-SERVICE-CLIENT] Client creation failed: {repr(e)}",
            flush=True,
        )

    # =========================================================================
    # Initialize XformPrim handles for people (for visual position updates)
    # =========================================================================
    try:
        from omni.isaac.core.prims import XFormPrim
        env._people_xforms = []
        for prim_path in _eval_config['people_prims']:
            try:
                xform = XFormPrim(prim_path)
                env._people_xforms.append(xform)
            except Exception as e:
                print(f"[EVAL] Warning: Could not create XFormPrim for {prim_path}: {e}")
                env._people_xforms.append(None)
        print(f"[EVAL] Initialized {len(env._people_xforms)} people XFormPrim handles")
    except Exception as e:
        print(f"[EVAL] Warning: Could not init people XFormPrims: {e}")
        env._people_xforms = []

    # =========================================================================
    # SECTION 4: Set up LiDAR to also cast against warehouse geometry
    # =========================================================================
    # The LiDAR raycaster was configured in NavigationEnv.__init__ with
    # mesh_prim_paths=["/World/ground"]. We need it to also cast against
    # the warehouse geometry at /World/Warehouse.
    #
    # Unfortunately, RayCaster's mesh_prim_paths is set at construction time
    # and may not be easily modifiable. We'll try to update it.
    # =========================================================================
    try:
        # Try to add warehouse to the raycaster's mesh list
        # This depends on the RayCaster implementation
        if hasattr(env.lidar, 'cfg'):
            print(f"[EVAL] Current LiDAR mesh_prim_paths: {env.lidar.cfg.mesh_prim_paths}")
            # The raycaster may need to be re-initialized with new mesh paths
            # For now, the warehouse floor should overlap with /World/ground
            # so static obstacles (shelves, walls) need to be in the mesh list
    except Exception as e:
        print(f"[EVAL] Note: Could not inspect LiDAR config: {e}")

    # =========================================================================
    # SECTION 5: Build the controller and policy (same as eval.py)
    # =========================================================================
    print("[EVAL] Building controller and policy...")

    ctrl = LeePositionController(9.81, env.drone.params).to(cfg.device)
    transformed_env = TransformedEnv(
        env,
        Compose(VelController(ctrl, yaw_control=False))
    ).train()
    transformed_env.set_seed(cfg.seed)

    # Build PPO agent and load checkpoint
    policy = PPO(
        cfg.algo,
        transformed_env.observation_spec,
        transformed_env.action_spec,
        cfg.device
    )

    print(f"[EVAL] Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=cfg.device)
    policy.load_state_dict(state_dict)
    print("[EVAL] Checkpoint loaded successfully")

    # Used when hovering is needed
    gated_policy = StartupGatedPolicy(
        base_policy=policy,
        env=env,
    )

    # =========================================================================
    # SECTION 6: Run evaluation loop
    # =========================================================================
    print("[EVAL] Starting evaluation...")
    print(f"[EVAL]   Warehouse: {warehouse_usd}")
    print(f"[EVAL]   People: {num_people}")
    print(f"[EVAL]   Checkpoint: {ckpt_path}")
    print(f"[EVAL]   Action limit: {cfg.algo.actor.action_limit} m/s")

    # Use the same evaluation approach as eval.py — SyncDataCollector
    # runs the policy in the environment for a fixed number of steps
    collector = SyncDataCollector(
        transformed_env,
        policy=policy,
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num,
        total_frames=cfg.env.num_envs * cfg.algo.training_frame_num * 10000,
        device=cfg.device,
        return_same_td=True,
        exploration_type=ExplorationType.MEAN,  # Deterministic eval
    )
    # Used when hovering is needed
    '''
    collector = SyncDataCollector(
        transformed_env,
        policy=gated_policy,
        frames_per_batch=(
            cfg.env.num_envs
            * cfg.algo.training_frame_num
        ),
        total_frames=(
            cfg.env.num_envs
            * cfg.algo.training_frame_num
            * 10000
        ),
        device=cfg.device,
        return_same_td=True,
        exploration_type=ExplorationType.MEAN,
    )
    '''
    print("[EVAL] Running policy... (Ctrl+C to stop)")
    try:
        _last_eval_wall_t = time.perf_counter()

        for i, data in enumerate(collector):
            _eval_wall_t = time.perf_counter()
            _eval_loop_dt_ms = (_eval_wall_t - _last_eval_wall_t) * 1000.0
            _last_eval_wall_t = _eval_wall_t
            '''
            if i % 5 == 0:
                print(f"[PROFILE-EVAL-LOOP] iter_dt={_eval_loop_dt_ms:.2f} ms", flush=True)
            '''
            #rclpy.spin_once(env.navrl_imu_ros_node, timeout_sec=0.0)

            # Move visual-only target marker to current NavRL target
            try:
                from pxr import UsdGeom, Gf
                stage = prim_utils.get_current_stage()
                marker = stage.GetPrimAtPath(env._target_marker_path)

                if marker.IsValid():
                    x = float(env.target_pos[0, 0, 0].detach().cpu())
                    y = float(env.target_pos[0, 0, 1].detach().cpu())
                    z = float(env.target_pos[0, 0, 2].detach().cpu())

                    xform = UsdGeom.Xformable(marker)
                    xform.ClearXformOpOrder()
                    xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))

            except Exception:
                pass

            # Print stats if available
            if 'next' in data.keys() and 'stats' in data['next'].keys():
                s = data['next']['stats']
                stats_str = []
                for k in ['reach_goal', 'collision', 'episode_len', 'return']:
                    if k in s.keys():
                        stats_str.append(f"{k}={s[k].float().mean().item():.3f}")
                if stats_str:
                    print(f"[EVAL step {i}] {' | '.join(stats_str)}", flush=True)
    except KeyboardInterrupt:
        print("\n[EVAL] Interrupted by user")

    # =========================================================================
    # Cleanup
    # =========================================================================
    wandb.finish()
    sim_app.close()
    print("[EVAL] Done.")


if __name__ == '__main__':
    main()
