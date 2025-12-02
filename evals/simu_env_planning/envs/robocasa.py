import logging
import os
import re
import sys
import time

import gym
import numpy as np
import robosuite
from robocasa.utils.dataset_registry import MULTI_STAGE_TASK_DATASETS, SINGLE_STAGE_TASK_DATASETS
from robocasa.utils.env_utils import create_env
from scipy.spatial.transform import Rotation as R

from evals.simu_env_planning.envs.wrappers.time_limit import TimeLimit

BASE_ASSET_ROOT_PATH = os.path.join(
    os.environ.get("JEPA_HOME", os.path.expanduser("~")),
    "robocasa/robocasa/models/assets/objects"
)

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

RCASA_CONTROLLER_INPUT_LIMS = np.array([1.0, -1])
RCASA_CONTROLLER_OUTPUT_LIMS = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0])

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class RoboCasaWrapper(gym.Wrapper):
    """
    Wrapper for RoboCasa environments.
    """

    def __init__(self, env, cfg=None, env_name="PnPCounterToSink", camera_name="robot0_agentview_left"):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.rescale_act_droid_to_rcasa = cfg.task_specification.env.get("rescale_act_droid_to_rcasa", False)
        logger.info(f"RoboCasaWrapper: {self.rescale_act_droid_to_rcasa=}")
        self.custom_task = cfg.task_specification.env.get("custom_task", False)
        self.subtask = cfg.task_specification.env.get("subtask", None)
        self.goal_obj_pos = None
        self.env_name = env_name
        self.camera_name = camera_name  # default camera name working with the underlying robosuite env
        self.custom_camera_name = self.camera_name
        self.camera_width = self.env.camera_widths[0]
        self.camera_height = self.env.camera_heights[0]
        self.full_action_dim = self.env.action_dim  # 12: 7 for arm, # 5 for base navigation
        self.manip_only = cfg.task_specification.env.get("manip_only", True)
        self.action_dim = 7 if self.manip_only else self.full_action_dim
        self.action_space = gym.spaces.Box(
            low=np.full(self.action_dim, -1.0), high=np.full(self.action_dim, 1.0), dtype=np.float32
        )
        self.reach_threshold = cfg.task_specification.env.get("reach_threshold", 0.2)
        self.place_threshold = cfg.task_specification.env.get("place_threshold", 0.15)
        logger.info(f"Set {self.reach_threshold=} and {self.place_threshold=}")
        if self.custom_task:
            self.custom_camera_name = "robot0_droid_agentview_left"  # "robot0_leftview"
            self.custom_camera_pos = ([0.4, 0.4, 0.6],)
            # [0.1, 0.4, 0.8]  # Example position [x, y, z]
            self.custom_camera_quat = [0.0, -0.0, 0.6, 1.0]
            # [0., -0.2, 0.6, 1.]
            # [0., 0.2, 0.6, 1.] not bad but looking a bit behind
            # [0., 0.2, 0., 1.]  # looks down a bit behind the robot
            # [0., -0.2, 0., 1.]: looks down a bit upwards in x direction
            # [0., -0.2, 0.866, 1.] : not bad but looks a bit high
            # w = cos(angle/2) = 0.5 (cos of 60°), y = sin(angle/2) = 0.866 (sin of 60°)
            # [0.42, 0.28, -0.48, -0.72]  # Example quaternion [w, x, y, z]
            self.custom_camera_fovy = 85

    # To avoid bug when wrapping in TimeLimit
    @property
    def spec(self):
        return None

    def eef_quat_to_xyz(self, eef_quat):
        # shape (4,)
        # If your quaternion is [w, x, y, z], convert to [x, y, z, w] for scipy
        eef_quat_xyzw = np.array([eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]])
        # Convert to Euler angles (xyz order, radians)
        eef_euler = R.from_quat(eef_quat_xyzw).as_euler("xyz", degrees=False)
        return eef_euler  # shape (3,)

    def gripper_2d_to_1d(self, gripper_qpos):
        """
        Convert 2D gripper position to 1D representation.
        Args:
            gripper_qpos: tensor of shape (2,) for gripper position
        Returns:
            tensor of shape (1,) for gripper state
        """
        return gripper_qpos[0:1] - gripper_qpos[1:2]

    def get_obs_proprio_succ_from_info(self, info):
        """
        Eitherway the obs part is not being used here, the only way for visual data to reach the PixelWrapper is
        via the self.render() function.
        """
        obs = np.random.randn(1)  # Dummy observation, not used
        # info[f'{self.camera_name}_image'] # H W 3
        eef_angle = self.eef_quat_to_xyz(info["robot0_eef_quat"])  # Convert quaternion to Euler angles
        gripper_closure = self.gripper_2d_to_1d(info["robot0_gripper_qpos"])  # Gripper position (2,) to closure (1,)
        info["proprio"] = np.concatenate(
            [
                info["robot0_eef_pos"],  # Cartesian position of the end effector (3,)
                eef_angle,  # Euler angles of the end effector (3,)
                gripper_closure,  # Gripper state (1,)
            ]
        )
        # Need to call this function to define env.obj_up_once
        # and other variables used in subtask_success()
        info["success"] = self.env._check_success()
        if self.subtask is not None:
            info = self.subtask_success(info)
            # info['success'] = self.subtask_success()
        # else:
        #     info = self.env._check_success(info)
        # info['success'] = self.env._check_success()
        return obs, info

    def subtask_success(self, info):
        obj = self.env.objects["obj"]
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])
        hand_pos = np.array(
            self.sim.data.body_xpos[self.sim.model.body_name2id(self.robots[0].gripper["right"].root_body)]
        )
        hand_obj_dist = np.linalg.norm(hand_pos - obj_pos)
        reach = hand_obj_dist < self.reach_threshold
        # We set goal_obj_pos after having reset the environment
        obj_goal_dist = np.linalg.norm(self.goal_obj_pos - obj_pos) if self.goal_obj_pos is not None else -1.0
        place = obj_goal_dist < self.place_threshold
        if self.subtask == "reach-pick-place":
            success = place
        elif self.subtask == "reach-pick":
            success = reach and self.env.obj_up_once
        elif self.subtask == "pick-place":
            success = self.env.obj_up_once and place
        elif self.subtask == "reach":
            success = reach
        elif self.subtask == "pick":
            success = self.env.obj_up_once
        elif self.subtask == "place":
            success = place
        else:
            raise ValueError(f"Unknown subtask: {self.subtask}")

        info["success"] = success
        info["obj_pos"] = obj_pos
        info["hand_pos"] = hand_pos
        info["obj_goal_dist"] = obj_goal_dist
        info["hand_obj_dist"] = hand_obj_dist
        info["obj_initial_height"] = self.env.obj_initial_height if hasattr(self.env, "obj_initial_height") else -1
        info["obj_lift"] = obj_pos[2] - info["obj_initial_height"]
        info["near_object"] = hand_obj_dist
        info["obj_up_once"] = self.env.obj_up_once if hasattr(self.env, "obj_up_once") else -1
        return info

    def reset(self, **kwargs):
        """
        Reset the environment and return the initial observation.
        """
        info = self.env.reset()
        return self.get_obs_proprio_succ_from_info(info)

    def reward(self):
        total_reward = 0.0
        if self.subtask == "reach":
            reached = self.env.reached_target
        elif self.subtask == "lift":
            reward = 0.3 if self.env.obj_up_once else 0.0

    def step(self, action):
        """
        Perform a step in the environment.
        action: np array of shape (action_dim,)
        """
        if self.manip_only:
            # If we're only controlling the arm, pad the action with zeros for the base nav
            full_action = np.zeros(self.full_action_dim)
            full_action[:7] = action  # First 7 dimensions are for arm control
            action = full_action
        scaled_action = full_action.copy()
        # scale to [-1, 1] expected range
        if self.rescale_act_droid_to_rcasa:
            scaled_action[:7] = action[:7] * RCASA_CONTROLLER_INPUT_LIMS[0] / RCASA_CONTROLLER_OUTPUT_LIMS
        info, reward, done, _ = self.env.step(scaled_action)
        obs, info = self.get_obs_proprio_succ_from_info(info)
        if info["success"]:
            logger.info("RoboCasaWrapper: Task success detected in step()")
        return obs, reward, None, done, info

    def render(self, *args, **kwargs):
        """
        Render the environment using the specified camera.
        Returns: H W 3
        Making a deepcopy is essential to avoid race conditions or corrupted images
        when the underlying simulator updates the visual buffer asynchronously
        """
        if self.custom_camera_name in self.env.sim.model._camera_name2id.keys():
            camera_to_use = self.custom_camera_name
        else:
            camera_to_use = self.camera_name
        logger.info(f"Using camera: {camera_to_use}")
        result = self.env.sim.render(
            height=self.camera_height, width=self.camera_width, camera_name=camera_to_use
        ).copy()
        if camera_to_use != "robot0_rightview":
            result = result[::-1]  # flip vertically
        else:
            # flip horizontally
            result = result[:, ::-1]
        return result

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def all_tasks(self):
        """
        Return all tasks available in the RoboCasa environment.
        """
        return list(SINGLE_STAGE_TASK_DATASETS.keys()) + list(MULTI_STAGE_TASK_DATASETS.keys()) + ["PnPCounterTop"]

    def update_env(self, env_info):
        pass

    def prepare(self, seed, init_state, env_info=None):
        """
        Inspired from robocasa/robocasa/utils/robomimic/robomimic_env_wrapper.py
        And updated with run_on_jimmy_mac branch of robocasa-murp/robocasa/scripts/playback_utils.py::reset_to()
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        prep_start_time = time.time()
        self.seed(seed)
        model_xml = env_info.get("model_xml", None)
        ep_meta = env_info.get("ep_meta", None)
        # # Uncomment to save out the model XML for debugging
        xml_path = f"evals/simu_env_planning/envs/robocasa/{self.env_name}_model.xml"
        with open(xml_path, "w") as f:
            f.write(model_xml)
            logger.info(f"Saved model XML to {xml_path}")
        if self.custom_task:
            # Modify the XML to add the custom camera
            import xml.etree.ElementTree as ET

            tree = ET.ElementTree(ET.fromstring(model_xml))
            camera_container = tree.find(".//body[@name='base0_support']")
            # for child in camera_container:
            #     logger.info(f"- {child.tag} with attributes: {child.attrib}")
            # worldbody = tree.find(".//worldbody")

            # Add the custom camera
            # camera_elem = ET.SubElement(worldbody, "camera")
            camera_elem = ET.SubElement(camera_container, "camera")
            camera_elem.set("name", self.custom_camera_name)
            camera_elem.set("pos", " ".join(map(str, self.custom_camera_pos)))
            camera_elem.set("quat", " ".join(map(str, self.custom_camera_quat)))
            camera_elem.set("fovy", str(self.custom_camera_fovy))
            camera_elem.set("mode", "fixed")

            # Convert the modified XML back to a string
            model_xml = ET.tostring(tree.getroot(), encoding="unicode")
            # custom_xml_path = f'evals/simu_env_planning/envs/robocasa/custom_envs/{self.env_name}_model.xml'
            # try:
            #     with open(custom_xml_path, "r") as f:
            #         model_xml = f.read()
            # self.custom_camera_name = "robot0_droid_agentview_left"
            #     logger.info(f"Loaded custom model XML from {custom_xml_path}")
            # except FileNotFoundError:
            #     logger.info(f"Warning: Custom XML file not found at {custom_xml_path}, using default XML")
        # First handle model reset if model_xml is provided
        if model_xml is not None:
            # Set episode metadata if provided
            if ep_meta is not None:
                # filter xml file to make sure asset paths do not point to jimmyyang path
                # like '/Users/jimmytyyang/research/robot-skills-sim/robocasa-murp/robocasa/models/assets/objects/aigen_objs/boxed_food/boxed_food_4/model.xml'
                ep_meta["object_cfgs"] = update_mjcf_paths(ep_meta["object_cfgs"])
                # Once filtere, prepare env for reset with this xml
                if hasattr(self.env, "set_attrs_from_ep_meta"):
                    self.env.set_attrs_from_ep_meta(ep_meta)
                elif hasattr(self.env, "set_ep_meta"):
                    self.env.set_ep_meta(ep_meta)

            # Reset the environment (without clearing ep_meta if we just set it)
            unset_ep_meta = ep_meta is None
            obs, info = self.reset()

            # xml = _prepare_xml(self.env, state["model"])

            # Process the model XML based on robosuite version
            # try:
            logger.info("Resetting from provided model XML")
            robosuite_version_id = int(robosuite.__version__.split(".")[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml

                xml = postprocess_model_xml(model_xml)
            else:
                # v1.4 and above use the class-based edit_model_xml function
                xml = self.env.edit_model_xml(model_xml)
            xml = path_change(xml)
            # Reset from XML string
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
            logger.info("Finished resetting from provided model XML")
            # except Exception as e:
            #     logger.info(f"Warning: Failed to reset from model XML: {e}")
        else:
            # Otherwise use standard reset
            obs, info = self.reset()

        # Some robosuite/robocasa environments have sim that can set state
        # if hasattr(self.env, 'sim') and hasattr(self.env.sim, 'set_state_from_flattened'):
        try:
            self.env.sim.set_state_from_flattened(init_state)
            self.env.sim.forward()

            # Update state as needed
            if hasattr(self.env, "update_sites"):
                # older versions of environment had update_sites function
                self.env.update_sites()
            if hasattr(self.env, "update_state"):
                # later versions renamed this to update_state
                self.env.update_state()

            # Get updated observation
            if hasattr(self.env, "_get_observation"):
                obs = self.env._get_observation()
            elif hasattr(self.env, "_get_observations"):
                obs = self.env._get_observations(force_update=True)
        except Exception as e:
            logger.info(f"Warning: Failed to set simulator state: {e}")
        logger.info(f"robocasa env.prepare() took {time.time() - prep_start_time:.2f} seconds")
        return obs, info

    @property
    def unwrapped(self):
        return self.env


def make_env(cfg):
    """
    Create a RoboCasa environment and wrap it with RoboCasaWrapper.
    """
    env_name = cfg.task_specification.task.split("-", 1)[-1]
    all_tasks = list(SINGLE_STAGE_TASK_DATASETS.keys()) + list(MULTI_STAGE_TASK_DATASETS.keys()) + ["PnPCounterTop"]
    if not cfg.task_specification.task.startswith("robocasa-") or env_name not in all_tasks:
        raise ValueError("Unknown task:", cfg.task_specification.task)
    # Dummy env that is later modified in RobocasaWrapper.prepare()
    # logger.info(f"Creating dummy RoboCasa PnPSinkToCounter..")
    env = create_env(
        env_name=env_name,  # e.g.,
        # "PnPSinkToCounter",
        robots=cfg.task_specification.env.get("robots", "PandaOmron"),
        camera_names=["robot0_leftview"],
        # ["robot0_agentview_left"],
        camera_widths=cfg.task_specification.img_size,
        camera_heights=cfg.task_specification.img_size,
        seed=cfg.meta.seed,
        render_onscreen=False,
    )
    # logger.info(f"Created dummy RoboCasa PnPSinkToCounter")
    env = RoboCasaWrapper(
        env, cfg, env_name, camera_name=cfg.task_specification.env.get("camera_name", "robot0_agentview_left")
    )
    logger.info(f"Wrapped RoboCasa environment with RoboCasaWrapper")
    env = TimeLimit(env, max_episode_steps=cfg.task_specification.max_episode_steps)
    env.max_episode_steps = env._max_episode_steps
    return env


def update_mjcf_paths(object_cfgs):
    """
    Update mjcf_path in object_cfgs by replacing src path with target path.

    Args:
        object_cfgs (list): list of object configuration dicts containing 'info' with 'mjcf_path'.
        src (str): source path substring to replace.
        target (str): target path substring to replace with.

    Returns:
        list: Updated object_cfgs with modified mjcf_path.
    """
    for i, object_cfg in enumerate(object_cfgs):
        path = object_cfg["info"]["mjcf_path"]
        models_index = path.find("objects")
        relative_path = path[models_index:]  # e.g. 'models/assets/objects/aigen_objs/apple/apple_5/model.xml'
        full_local_path = os.path.join(BASE_ASSET_ROOT_PATH, relative_path[len("objects/") :])
        object_cfgs[i]["info"]["mjcf_path"] = full_local_path
    return object_cfgs


def path_change(xml_string):
    """
    Fix absolute file paths in the MJCF XML by replacing them with local paths
    rooted at BASE_ASSET_ROOT_PATH.
    """

    def replace_path(match):
        original_path = match.group(1)
        model_index = original_path.find("objects/")
        if model_index == -1:
            return f'file="{original_path}"'

        relative_path = original_path[model_index + len("objects/") :]
        new_path = os.path.join(BASE_ASSET_ROOT_PATH, relative_path)
        new_path = os.path.normpath(new_path)

        return f'file="{new_path}"'

    updated_xml = re.sub(r'file="([^"]+)"', replace_path, xml_string)
    return updated_xml


def _prepare_xml(env, model_xml):
    robosuite_version_id = int(robosuite.__version__.split(".")[1])
    if robosuite_version_id <= 3:
        from robosuite.utils.mjcf_utils import postprocess_model_xml

        return postprocess_model_xml(model_xml)
    else:
        return env.edit_model_xml(model_xml)
