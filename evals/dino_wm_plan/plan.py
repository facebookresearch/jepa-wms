import os
import sys
import gym
import hydra
import random
import torch
import pickle
import warnings
import numpy as np
from pathlib import Path
from einops import rearrange, repeat
from omegaconf import OmegaConf, open_dict
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from evals.dino_wm_plan.env.venv import SubprocVectorEnv
from evals.dino_wm_plan.custom_resolvers import replace_slash

# from .traj_dset import TrajDataset
from evals.dino_wm_plan.preprocessor import Preprocessor
from evals.dino_wm_plan.planning.evaluator import PlanEvaluator
from evals.dino_wm_plan.utils import (
    cfg_to_dict,
    seed,
    slice_trajdict_with_t,
    aggregate_dct,
    move_to_device,
    concat_trajdict,
)
import wandb
import logging
import json

log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
    "predictors",
]

import submitit
from itertools import product
import importlib


def planning_main_in_dir(working_dir, cfg_dict):
    # log.info(f"Before chdir: {os.getcwd()=}") # does not print anything anywhere
    os.chdir(working_dir)
    # log.info(f"After chdir: {os.getcwd()=}")
    return planning_main(cfg_dict=cfg_dict)


def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            # log.info(f"in launch_plan_jobs: {os.getcwd()=}") # printed once per thread
            # log.info(f"{subdir_path=}")
            # os.chdir('/home/basileterv/dino_wm_pub_fork')  # Update this path
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            sys.path.append('/home/basileterv/dino_wm_pub_fork')
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=["gd", "cem"],
    goal_source=["dset"],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
    model_type="dino_wm",
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    from hydra import initialize, compose

    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["model_type"] = model_type
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        # have different seeds for each planning instances
        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )

        # tmp: assume planning horizon equals to goal horizon
        from evals.dino_wm_plan.planning.mpc import MPCPlanner

        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict["goal_H"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []

        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)
            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.stack([obs_0[k]] * self.wm.tubelet_size, axis=1)
                obs_g[k] = np.stack([obs_g[k]] * self.wm.tubelet_size, axis=1)
                # obs_0[k] = np.expand_dims(obs_0[k], axis=1).repeat(1, self.wm.tubelet_size)
                # obs_g[k] = np.expand_dims(obs_g[k], axis=1).repeat(1, self.wm.tubelet_size)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + self.wm.tubelet_size)
            )
            # Save initial and goal observations as images
            # plt.figure(figsize=(10, 5))
            # # Plot initial observation
            # plt.subplot(1, 2, 1)
            # plt.imshow(observations[-1]["visual"][0].permute(1, 2, 0))
            # plt.title("Initial Observation")
            # plt.axis('off')
            # # Plot goal observation
            # plt.subplot(1, 2, 2)
            # plt.imshow(observations[-1]["visual"][-1].permute(1, 2, 0))
            # plt.title("Goal Observation")
            # plt.axis('off')
            # # Save plot to PDF
            # plt.savefig(f"/home/basileterv/dino_wm_pub_fork/plots/sample_traj_obs.pdf", bbox_inches='tight')
            # plt.close()

            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: arr[:, :self.wm.tubelet_size]
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: arr[:, -self.wm.tubelet_size:]
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions
        # observations = aggregate_dct(observations)
        # self.obs_0 = {key: arr[:, :1, ...] for key, arr in observations.items()}
        # self.obs_g = {key: arr[:, -1:, ...] for key, arr in observations.items()}

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            print("traj id: ", traj_id, "  Offset: ", offset) # TODO: delete after check
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        file_path = os.path.abspath("plan_targets.pkl")
        print(f"Dumped plan targets to {file_path}")

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_ckpt_jepa(train_cfg, device):
    # module_name = "models.vjepa_droid_v4.modelcustom.vit_enc_preds"
    encoder, predictors = importlib.import_module(f"{train_cfg.module_name}").init_module(
        model_kwargs=dict(train_cfg.pretrain_kwargs),
        device=device,
        img_size=224,
        pretrained_encoder_arch=True,
        pretrained_predictor_arch=True,
        pretrained_encoder_weights=True,
        pretrained_predictor_weights=True,
    )
    return {"encoder": encoder, "predictor": predictors[0], "epoch": 150}


def load_model(model_ckpt, train_cfg, num_action_repeat, device, model_type):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device) if model_type == "dino_wm" else load_ckpt_jepa(train_cfg, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        # Create encoder from config - use the path in the current codebase
        # Original DINO-WM uses models.dino.DinoV2Encoder, but in this codebase
        # the encoder is at app/plan_common/models/dino.DinoEncoder
        from app.plan_common.models.dino import DinoEncoder
        encoder_cfg = train_cfg.encoder
        encoder_name = getattr(encoder_cfg, 'name', 'dinov2_vits14')
        result["encoder"] = DinoEncoder(
            name=encoder_name,
            feature_key="x_norm_patchtokens",
        ).to(device)
        for p in result["encoder"].parameters():
            p.requires_grad = False
        result["encoder"] = result["encoder"].eval()
        print(f"Created DinoEncoder with name={encoder_name}")

    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    # Create VWorldModel directly instead of using hydra instantiate
    # Original DINO-WM uses models.visual_world_model.VWorldModel, but in this codebase
    # the VWorldModel is at evals.dino_wm_plan.visual_world_model
    from evals.dino_wm_plan.visual_world_model import VWorldModel

    model = VWorldModel(
        image_size=train_cfg.model.image_size,
        num_hist=train_cfg.model.num_hist,
        num_pred=train_cfg.model.num_pred,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"] if "proprio_encoder" in result.keys() else None,
        action_encoder=result["action_encoder"] if "action_encoder" in result.keys() else None,
        predictor=result["predictor"],
        decoder=result["decoder"] if "decoder" in result.keys() else None,
        proprio_dim=train_cfg.proprio_emb_dim if model_type == "dino_wm" else 0,
        action_dim=train_cfg.action_emb_dim if model_type == "dino_wm" else train_cfg.pretrain_kwargs.predictor.cfgs_model.pred_embed_dims[0],
        concat_dim=train_cfg.concat_dim if model_type == "dino_wm" else 1,
        num_action_repeat=num_action_repeat if model_type == "dino_wm" else 1,
        num_proprio_repeat=train_cfg.num_proprio_repeat if model_type == "dino_wm" else 1,
        train_encoder=train_cfg.model.get("train_encoder", False),
        train_predictor=train_cfg.model.get("train_predictor", False),
        train_decoder=train_cfg.model.get("train_decoder", False),
    )
    model.to(device)
    return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    if cfg_dict["model_type"] == "dino_wm":
        model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}"
        with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
            model_cfg = OmegaConf.load(f)
        seed(cfg_dict["seed"])
        # model_cfg.env already includes config to build the eval env
        model_cfg.env.dataset["traj_subset"] = True
        _, dset = hydra.utils.call(
            model_cfg.env.dataset,
            num_hist=model_cfg.num_hist,
            num_pred=model_cfg.num_pred,
            frameskip=model_cfg.frameskip,
        )
        dset = dset["valid"]

        num_action_repeat = model_cfg.num_action_repeat
        model_ckpt = (
            Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
        )
    elif cfg_dict["model_type"] == "vjepa_droid_v4":
        model_path = f"{ckpt_base_path}/{cfg_dict['model_name']}"
        with open(os.path.join(model_path, "base_reach_100.yaml"), "r") as f:
            model_cfg = OmegaConf.load(f).model_kwargs
        # ---------
        model_cfg["model"] = {}
        model_cfg["model"]["_target_"] = "models.vjepa_world_model.VWorldModel"
        model_cfg["model"]["image_size"] = 224
        model_cfg["model"]["num_hist"] = 2
        model_cfg["num_hist"] = 2
        model_cfg["model"]["num_pred"] = 1
        model_cfg["num_pred"] = 1
        model_cfg["model"]["train_encoder"] = False
        model_cfg["model"]["train_predictor"] = True
        model_cfg["model"]["train_decoder"] = False
        model_cfg["has_decoder"] = False
        model_cfg["frameskip"] = 5

        model_cfg.env = {
            'name': 'point_maze',
            'args': [],
            'kwargs': {},
            'dataset': {'_target_': 'datasets.point_maze_dset.load_point_maze_slice_train_val',
                        'n_rollout': None,
                        'normalize_action': True,
                        'data_path': '/checkpoint/amaia/video/basileterv/dino_wm/data/point_maze',
                        'split_ratio': 0.9,
                        'transform': {'_target_': 'datasets.img_transforms.default_transform', 'img_size': 224}
                        },
            # 'decoder_path': None,
            # 'num_workers': 16,
            }
        # -----------
        seed(cfg_dict["seed"])
        _, dset = hydra.utils.call(
            model_cfg.env.dataset,
            num_hist=model_cfg.num_hist,
            num_pred=model_cfg.num_pred,
            frameskip=model_cfg.frameskip,
        )
        dset = dset["valid"]
        num_action_repeat = 1
        model_ckpt = Path(model_cfg.pretrain_kwargs.folder) / f"jepa-latest.pth.tar"

    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device, model_type=cfg_dict["model_type"])
    # use dummy vector env for wall and deformable envs
    if model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from evals.dino_wm_plan.env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )

    logs = plan_workspace.perform_planning()
    return logs


@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        # cfg_dict["hydra"]["run"] is plan_outputs/${now:%Y%m%d%H%M%S}_${replace_slash:${model_name}}_gH${goal_H}
        # hence os.getcwd() is like /storage/home/basileterv/dino_wm_pub_fork/plan_outputs/20250204153402_2025-01-21_15-08-08_gH5
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    # cfg_dict["wandb_logging"] = True
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
