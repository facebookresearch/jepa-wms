# The below code is inspired from TD-MPC2 https://github.com/nicklashansen/tdmpc2
# licensed under the MIT License

import warnings
from copy import deepcopy

import gym

from evals.simu_env_planning.envs.wrappers.multitask import MultitaskWrapper
from evals.simu_env_planning.envs.wrappers.pixels import PixelWrapper
from evals.simu_env_planning.envs.wrappers.tensor import TensorWrapper


def missing_dependencies(task):
    raise ValueError(f"Missing dependencies for task {task}; install dependencies to use this environment.")


from evals.simu_env_planning.envs.droid_dset_dummy_env import make_env as make_droid_dset_dummy_env
from evals.simu_env_planning.envs.pusht_gym_wrap import make_env as make_pusht_env
from evals.simu_env_planning.envs.wall_gym_wrap import make_env as make_wall_env

try:
    from evals.simu_env_planning.envs.pointmaze_gym_wrap import make_env as make_maze_env
except ImportError as e:
    make_maze_env = missing_dependencies
try:
    from evals.simu_env_planning.envs.robocasa import make_env as make_robocasa_env
except ImportError as e:
    make_robocasa_env = missing_dependencies
try:
    from evals.simu_env_planning.envs.metaworld import make_env as make_metaworld_env
except Exception as e:
    make_metaworld_env = missing_dependencies


warnings.filterwarnings("ignore", category=DeprecationWarning)


def make_multitask_env(cfg):
    """
    Make a multi-task environment. Only used for Metaworld.
    """
    print("Creating multi-task environment with tasks:", cfg.tasks)
    envs = []
    for task in cfg.tasks:
        _cfg = deepcopy(cfg)
        _cfg.task = task
        _cfg.task_specification.multitask = False
        env = make_env(_cfg)
        if env is None:
            raise ValueError("Unknown task:", task)
        envs.append(env)
    env = MultitaskWrapper(cfg, envs)
    cfg.obs_shapes = env._obs_dims
    cfg.action_dims = env._action_dims
    cfg.episode_lengths = env._episode_lengths
    return env


def make_env(cfg):
    """
    Flexible interface to build environments.
    """
    gym.logger.set_level(40)
    if cfg.task_specification.goal_source in ["dset", "random_action"]:
        if cfg.task_specification.task == "droid-base":
            cfg.task_specification.max_episode_steps = cfg.planner.horizon
        elif cfg.task_specification.task.startswith("robocasa"):
            pass
        else:  # pusht
            cfg.task_specification.max_episode_steps = cfg.frameskip * cfg.task_specification.goal_H
            cfg.task_specification.goal_max_episode_steps = cfg.frameskip * cfg.task_specification.goal_H
    elif cfg.task_specification.goal_source == "random_state":
        # TODO: Hardcoded for now, improve
        cfg.task_specification.max_episode_steps = cfg.frameskip * cfg.task_specification.goal_H
    else:
        if cfg.task_specification.get("max_episode_steps", None) is None:
            cfg.task_specification.max_episode_steps = 100

    if cfg.task_specification.multitask:
        env = make_multitask_env(cfg)

    else:
        env = None
        if cfg.task_specification.task.startswith("mw-"):
            env = make_metaworld_env(cfg)
        elif cfg.task_specification.task.startswith("pusht-"):
            env = make_pusht_env(cfg)
        elif cfg.task_specification.task.startswith("wall-"):
            env = make_wall_env(cfg)
        elif cfg.task_specification.task.startswith("maze-"):
            env = make_maze_env(cfg)
        elif cfg.task_specification.task.startswith("robocasa-"):
            env = make_robocasa_env(cfg)
        elif cfg.task_specification.task.startswith("droid-"):
            env = make_droid_dset_dummy_env(cfg)

        env = TensorWrapper(env)
        if cfg.task_specification.get("obs", "state") in ["rgb", "rgb_state"]:
            env = PixelWrapper(cfg, env)
    try:  # Dict
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except:  # Box
        cfg.obs_shape = {cfg.task_specification.get("obs", "state"): env.observation_space.shape}
    if cfg.task_specification.get("obs", "state") == "rgb_state":
        cfg.obs_shape = {"state": [4], "rgb": cfg.obs_shape["rgb_state"]}

    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.meta.seed_steps = max(1000, 5 * cfg.episode_length)
    return env
