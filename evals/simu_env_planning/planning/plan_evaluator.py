import os
from time import time
import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from tensordict.tensordict import TensorDict
from tqdm.auto import tqdm

from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.episode_plot_utils import (
    analyze_distances,
    compare_unrolled_plan_expert,
    plot_actions_comparison,
    plot_losses,
)
from evals.simu_env_planning.planning.utils import (
    make_td,
    make_video,
    make_video_pdf,
    prepare_obs,
    save_decoded_frames,
    save_init_goal_frame,
)

from src.utils.logging import get_logger

log = get_logger(__name__)


class PlanEvaluator:
    def __init__(self, cfg, agent):
        self.cfg = cfg
        self.agent = agent

    def get_goal_state_mw(self, env, task_idx=-1):
        """a function to copy the environment and run the expert
        policy to get the goal state.
        It also saves the images along the way.
        Cleaner variant where both envs get the same rand_vec thanks to set_task
        Recall that each env.reset() modifies the rand_vec randonmly. So each iteration
        of the loop in main() / call to eval() gives statistically independent environments.
        """
        from metaworld import _encode_task

        from evals.simu_env_planning.envs.metaworld import task_name_to_policy

        policy_cls = task_name_to_policy(self.cfg.tasks[task_idx])
        # we need to unwrap the env and deepcopy it
        unwrapped = env.proprio_env.unwrapped
        env_cls = type(unwrapped)
        rand_vec = unwrapped._last_rand_vec
        task_data = _encode_task(
            "mw-reach",  # any task works, metaworld only looks at env_cls
            {
                "rand_vec": rand_vec,
                "env_cls": env_cls.__bases__[0],
                "partially_observable": False,
            },
        )
        log.info("Creating expert env ...")
        # env_expert is a MultitaskWrapper if env is
        env_expert = make_env(env.cfg)
        # make sure the env_expert is set to the right task before set_task
        obs, info = env_expert.reset(task_idx=task_idx)
        # set_task is the only way to freeze the rand_vec
        env_expert.set_task(task_data)
        env.set_task(task_data)
        # Now need a second reset with the right rand_vec
        obs, info = env_expert.reset(task_idx=task_idx)
        obs = make_td(obs, info)
        policy = policy_cls()

        def actor(info, steps_left):
            # need to convert to tensor because the env expects a tensor
            # return torch.tensor(env_expert.action_space.sample())
            return torch.tensor(np.expand_dims(policy.get_action(info["state"]), axis=0))

        return self.unroll_expert(env_expert, obs, info, actor)

    def unroll_expert(self, env, obs, info, actor):
        """
        Returns:
        List of Tensordicts with length the episode length, each td has 2 fields: visual and proprio
        """
        done = False
        ep_reward = 0
        td = obs
        ep_obs_proprio_td_list = [td]
        infos_list = []
        actions = []
        pbar = tqdm(
            desc="executing expert",
            total=env.max_steps(),
            initial=env.elapsed_steps(),
            position=0,
            leave=True,
            disable=env.cfg.logging.tqdm_silent,
        )
        while not done:
            action = actor(
                info,
                steps_left=max((env.steps_left() + 1) // self.cfg.frameskip, 1),
            )
            actions.append(action.detach().cpu())
            obses, rewards, dones, infos = env.step_multiple(action)
            reward = sum(rewards)
            done = dones[-1]

            ep_reward += reward
            for obs, info in zip(obses, infos):
                td = make_td(obs, info)
                ep_obs_proprio_td_list.append(td)
                infos_list.append(info)
            pbar.update(len(obses))
            pbar.set_postfix(
                {
                    "near_object": infos[-1]["near_object"],
                    "success": infos[-1]["success"],
                }
            )
        pbar.close()
        return ep_obs_proprio_td_list, ep_reward, actions, infos_list

    def unroll_agent(self, env, obs, info, actor, preprocessor=None):
        """
        Returns:
        List of Tensordicts with length the episode length, each td has 2 fields: visual and proprio
        If "droid" in cfg.task_specification.task, the proprio and obs outputted by env.step_multiple() are dummy
            so should not replan on them, hence done = True after first call to agent_actor.
        """
        done = False
        ep_reward = 0
        td = obs
        ep_obs_proprio_td_list = [td]
        infos_list = []
        actions = []
        pbar = tqdm(
            desc="executing agent",
            total=env.max_steps(),
            initial=env.elapsed_steps(),
            position=0,
            leave=True,
            disable=env.cfg.logging.tqdm_silent,
        )
        while not done:
            plan_vis_path = (
                f"{self.ep_plan_vis_dir}/step{env.elapsed_steps()}" if self.cfg.planner.decode_each_iteration else None
            )
            act_time_start = time()
            # we always have action_skip 1 except for DROID -> robocasa
            # For DROID -> robocasa, we can repeat act skip or not
            if not self.cfg.planner.repeat_actskip:
                plan_steps_left = max((env.steps_left() + 1) * self.agent.model.action_skip // self.cfg.frameskip, 1)
            else:
                plan_steps_left = max((env.steps_left() + 1), 1)
            action = actor(
                td,
                steps_left=plan_steps_left,
            )
            act_time_end = time()
            log.info(f"Action optim at step {env.elapsed_steps()} took {act_time_end - act_time_start:.2f} seconds")
            if self.cfg.logging.optional_plots and self.prev_pred_frames_over_iterations[-1] is not None:
                save_decoded_frames(
                    self.prev_pred_frames_over_iterations[-1],
                    self.prev_losses[-1],
                    plan_vis_path=plan_vis_path,
                )
            action = rearrange(action.cpu(), "t (f d) -> (t f) d", d=env.action_dim)
            if self.cfg.planner.repeat_actskip:
                action = action.repeat_interleave(self.agent.model.action_skip, dim=0)
            if preprocessor is not None:
                action = preprocessor.denormalize_actions(action)
            actions.append(action.detach().cpu())
            # env.step_multiple() returns consequences of action
            # but not initial obs, stored in ep_obs_proprio_td_list[-1]["visual"]
            obses, rewards, dones, infos = env.step_multiple(action)
            reward = sum(rewards)
            done = dones[-1]

            # Plot stepped actions of this slice only if needed
            if (
                self.cfg.planner.decode_each_iteration
                and not "droid" in self.cfg.task_specification.task
                and self.cfg.logging.optional_plots
            ):
                agent_gt_path = f"{plan_vis_path}_gt.pdf"
                step_indices = np.arange(self.cfg.frameskip - 1, len(obses), self.cfg.frameskip)
                frames = [ep_obs_proprio_td_list[-1]["visual"].to(torch.uint8)] + [obses[i] for i in step_indices]
                # frames = [obses[0]] + obses[1:][::self.cfg.frameskip]
                n_frames = len(frames)
                fig, axes = plt.subplots(1, n_frames, figsize=(5 * n_frames, 5))
                if n_frames == 1:
                    axes = [axes]
                for ax, frame in zip(axes, frames):
                    if isinstance(frame, torch.Tensor):
                        frame = frame.detach().cpu().numpy()
                    frame = np.squeeze(frame, axis=0).transpose(1, 2, 0)
                    ax.imshow(frame)
                    ax.axis("off")
                plt.tight_layout()
                plt.savefig(agent_gt_path, format="pdf", bbox_inches="tight")
                plt.close()
                print(f"Last iteration frames saved to {agent_gt_path}")

            # Evaluate success
            if (
                any(pref in self.cfg.task_specification.task for pref in ["mw", "robocasa"])
                and self.cfg.task_specification.succ_def == "simu"
            ):
                success = infos[-1]["success"]
                if str(self.cfg.task_specification).startswith("robocasa-"):
                    state_dist = infos[-1]["hand_obj_dist"]
                else:
                    state_dist = np.linalg.norm(self.state_g - infos[-1]["state"])
            else:
                eval_results = env.eval_state(self.state_g, infos[-1]["state"])
                success = eval_results["success"]
                state_dist = eval_results["state_dist"]
            if success and self.cfg.task_specification.done_at_succ:
                done = True
            ep_reward += reward
            for obs, info in zip(obses, infos):
                td = make_td(obs, info)
                ep_obs_proprio_td_list.append(td)
                infos_list.append(info)
            pbar.update(len(obses))
            pbar.set_postfix(
                {
                    "near_object": infos[-1]["near_object"],
                    "obj_goal_dist": infos[-1].get("obj_goal_dist", -1.0),
                    "success": success,
                    "obj_lift": infos[-1].get("obj_lift", -1.0),
                }
            )
            log.info(f"obj_initial_height {infos[-1].get('obj_initial_height', -1.0)}")
            log.info(f"obj_up_once {infos[-1].get('obj_up_once', -1.0)}")
            # We cannot step actions so cannot replan in the DROID case
            if "droid" in self.cfg.task_specification.task:
                done = True

        pbar.close()
        return ep_obs_proprio_td_list, ep_reward, actions, infos_list, success, state_dist

    def sample_traj_segment_from_dset(
        self,
        cfg,
        agent,
        traj_len,
        droid=False,
        goal_last=False,
    ):
        log.info(f"Sampling trajectory segment of length {traj_len} from dataset")
        if droid:
            # log.info(f"{agent.local_generator.get_state()[:100]=}")
            ep_idx = torch.randint(low=0, high=len(agent.dset), size=(1,), generator=agent.local_generator).item()
            obs, act, state, e_info = agent.dset[ep_idx]
            # log.info(f"Sampled {ep_idx=}")
            max_offset = obs["visual"].shape[0] - traj_len
            offset = torch.randint(low=0, high=max_offset + 1, size=(1,), generator=agent.local_generator).item()
            # log.info(f"Sampled {offset=}")
            obs = {key: arr[offset : offset + traj_len] for key, arr in obs.items()}
            state = state[offset : offset + traj_len]
            act = act[offset : offset + cfg.frameskip * cfg.task_specification.goal_H]
        else:
            if cfg.task_specification.env.get(
                "sample_subtask_slice", False
            ) and cfg.task_specification.task.startswith("robocasa-"):
                max_attempts = 100  # Add a limit to prevent infinite loops
                attempt = 0
                sampled_traj = False

                while not sampled_traj and attempt < max_attempts:
                    attempt += 1
                    traj_id = torch.randint(
                        low=0, high=len(agent.dset), size=(1,), generator=agent.local_generator
                    ).item()
                    try:
                        obs, act, state, reward, e_info = agent.dset.__getitem__(
                            traj_id, subtask=cfg.task_specification.env.subtask
                        )
                        sampled_traj = True
                        log.info(
                            f"Sampled subtask {cfg.task_specification.env.subtask} from traj {traj_id} (attempt {attempt})"
                        )
                    except Exception as e:
                        log.info(
                            f"Failed to sample subtask {cfg.task_specification.env.subtask} from traj {traj_id}: {e}"
                        )
                        # No need to raise here - just continue the loop

                if not sampled_traj:
                    raise RuntimeError(
                        f"Failed to find a trajectory with subtask {cfg.task_specification.env.subtask} after {max_attempts} attempts"
                    )
            else:
                # Check if any trajectory is long enough
                valid_traj = [
                    agent.dset.get_seq_length(i)
                    for i in range(len(agent.dset))
                    if agent.dset.get_seq_length(i) >= traj_len
                ]
                if len(valid_traj) == 0:
                    raise ValueError("No trajectory in the dataset is long enough.")
                max_offset = -1
                while max_offset < 0:  # filter out traj that are not long enough
                    traj_id = torch.randint(
                        low=0, high=len(agent.dset), size=(1,), generator=agent.local_generator
                    ).item()
                    obs, act, state, reward, e_info = agent.dset[traj_id]
                    max_offset = obs["visual"].shape[0] - traj_len
                state = state.numpy()
                if goal_last:
                    offset = max_offset
                else:
                    offset = torch.randint(
                        low=0, high=max_offset + 1, size=(1,), generator=agent.local_generator
                    ).item()
                log.info(f"Sampled traj: traj id: {traj_id}  Offset: {offset}")
                obs = {key: arr[offset : offset + traj_len] for key, arr in obs.items()}
                state = state[offset : offset + traj_len]
                act = act[offset : offset + cfg.frameskip * cfg.task_specification.goal_H]
        return obs, state, act, e_info

    def set_episode(self, cfg, agent, env, ep_seed, task_idx=-1):
        """
        Sets up the initial and goal states for an evaluation episode,
        and potentially the expert trajectory and actions.
        """
        if cfg.task_specification.goal_source == "expert":
            # reset_warmup instead of env.reset seems to set the init_obs
            # corresponding to the rand_vec
            init_obs, info = env.reset_warmup(seed=ep_seed)
            init_obs = make_td(init_obs, info)
            if cfg.task_specification.task.startswith("mw-"):
                expert_obses, ep_reward, expert_actions, expert_infos = self.get_goal_state_mw(
                    env,
                    task_idx=task_idx,
                )
            self.expert_actions = torch.stack(expert_actions).permute(1, 0, 2)  # [1, T, act_dim]
            goal_obs = expert_obses[-1]  # obs: (t c h w), proprio (d)
            expert_success = expert_infos[-1]["success"]
            self.state_g = expert_infos[-1]["state"]
        elif cfg.task_specification.goal_source == "random_state":
            rand_init_state, rand_goal_state = env.sample_random_init_goal_states(ep_seed)
            # the order is essential, env.prepare sets the state of the env to its state arg
            goal_obs, goal_info = env.prepare(ep_seed, rand_goal_state)
            goal_obs = make_td(goal_obs, goal_info)
            init_obs, info = env.prepare(ep_seed, rand_init_state)
            init_obs = make_td(init_obs, info)
            expert_success = 1
            self.state_g = goal_info["state"]
            expert_obses = []
        elif cfg.task_specification.goal_source in ["dset", "random_action"]:
            # Sample a trajectory segment from the dataset, the goal and initial states are tensordicts
            # with a time dimension of num_frames and num_proprios, with adjacent (in time) frames and proprios
            # Because of env.reset_warmup() the num_frames initial frames and proprios are identical.
            assert "mw" not in cfg.task_specification.task
            # TODO: for metaworld, since we cannot prepare, we can sample_traj with full
            # length, step only some GT actions to get init_state, set the goal to a state at distance 25
            # from the dset, or by creating another env and stepping all actions to it.
            # But we cannot retrieve the seed or rand_vec that allowed to start the sampled traj segment.
            observations, states, actions, env_info = self.sample_traj_segment_from_dset(
                cfg,
                agent,
                traj_len=cfg.frameskip * cfg.task_specification.goal_H + 1,
                droid=("droid" in cfg.task_specification.task),
                goal_last=cfg.task_specification.get("goal_last", False),
            )
            if cfg.task_specification.env.get("subtask") is not None and "place" in cfg.task_specification.env.subtask:
                # CAREFUL: We directly modify the attribute of the wrapped RobocasaWrapper env
                env.env.env.env.goal_obj_pos = env_info["meta_data_info"]["current_obj_pos"][-1]
            env.update_env(env_info)

            init_state = states[0]
            init_state = np.array(init_state)
            if "droid" in cfg.task_specification.task:
                self.expert_actions = torch.tensor(actions).unsqueeze(0)
                expert_obses = []
                for i in range(len(observations["visual"])):
                    expert_obses.append(
                        TensorDict(
                            {
                                "visual": (
                                    self.agent.preprocessor.inverse_transform(observations["visual"][i : i + 1]) * 255
                                ).to(torch.uint8),
                                "proprio": torch.as_tensor(observations["proprio"][i : i + 1], dtype=torch.float32),
                            }
                        )
                    )
                self.state_g = states[-1]
            else:
                if cfg.task_specification.get("replay_expert", True):
                    if cfg.task_specification.goal_source == "random_action":
                        actions = torch.randn_like(actions, generator=agent.local_generator)
                    exec_actions = agent.preprocessor.denormalize_actions(actions)
                    self.expert_actions = actions.detach().clone().unsqueeze(0)
                    # replay actions in env to get expert_obses
                    env.set_max_steps(cfg.task_specification.goal_max_episode_steps)
                    env_roll_start_time = time()
                    expert_obses, rollout_infos = env.rollout(
                        ep_seed,
                        init_state,
                        exec_actions,
                        env_info,
                    )
                    log.info(f"Env rollout took {time() - env_roll_start_time:.2f} seconds")
                    expert_obses = [
                        make_td(obs, rollout_info) for (obs, rollout_info) in zip(expert_obses, rollout_infos)
                    ]
                    self.state_g = rollout_infos[-1]["state"]
                else:
                    # expert obses are just the dset ones
                    self.expert_actions = torch.tensor(actions).unsqueeze(0)
                    expert_obses = []
                    for i in range(len(observations["visual"])):
                        expert_obses.append(
                            TensorDict(
                                {
                                    "visual": torch.tensor(
                                        self.agent.preprocessor.inverse_transform(observations["visual"][i : i + 1])
                                        * 255,
                                        dtype=torch.uint8,
                                    ),
                                    "proprio": torch.tensor(observations["proprio"][i : i + 1], dtype=torch.float32),
                                }
                            )
                        )
                    self.state_g = states[-1]
                # Important: reprepare env back to same initial state
                reset_vis, reset_info = env.prepare(ep_seed, init_state, env_info=env_info)
                if "max_episode_steps" in cfg.task_specification:
                    env.set_max_steps(cfg.task_specification.max_episode_steps)
            init_obs = expert_obses[0]
            # since expert_obses is obtained by stepping all exec_actions, there is no skip and
            goal_obs = expert_obses[-1]
            expert_success = 1
        else:
            raise ValueError(f"Unknown goal source: {cfg.task_specification.goal_source}")
        return init_obs, goal_obs, expert_obses, expert_success

    def eval(self, cfg, agent, env, task_idx=-1, ep=0):
        """
        Central evaluation function called by the eval loop.
        """
        # Only the MultitaskWrapper has task attribute, hence using cfg.tasks[task_idx]
        # to account for single-task case
        work_dir = cfg.work_dir / cfg.tasks[task_idx] / f"ep_{ep}"  # / f"seed_{cfg.meta.seed}"
        dist_work_dir = work_dir / "distances"
        os.makedirs(dist_work_dir, exist_ok=True)
        vis_work_dir = work_dir / "visualisation"
        os.makedirs(vis_work_dir, exist_ok=True)
        if cfg.planner.decode_each_iteration:
            self.ep_plan_vis_dir = work_dir / "plan_vis"
            os.makedirs(self.ep_plan_vis_dir, exist_ok=True)
        ep_seed = (cfg.local_seed * cfg.local_seed + ep * cfg.local_seed) % (2**32 - 2)
        self.expert_actions = None
        # To set agent env to the correct task_idx if env is multitask
        # this changes the rand_vec since we had not frozen it.
        init_obs, info = env.reset(seed=ep_seed, task_idx=task_idx)
        # ensure different rand_vec between episodes
        unwrapped = env.proprio_env.unwrapped
        unwrapped._freeze_rand_vec = False
        # Ensure different seed accross episodes, see SawyerXYZEnv._get_state_rand_vec()
        unwrapped.seeded_rand_vec = True
        env.seed(ep_seed)

        init_obs, goal_obs, expert_obses, expert_success = self.set_episode(
            cfg,
            agent,
            env,
            ep_seed,
            task_idx=task_idx,
        )
        if cfg.logging.optional_plots:
            save_init_goal_frame(
                init_obs, goal_obs, vis_work_dir=vis_work_dir, concat_channels=env.obs_concat_channels
            )

        if goal_obs["visual"].shape[0] > agent.model.tubelet_size_enc:
            # Keep the minimal last state
            goal_obs["visual"] = goal_obs["visual"][-agent.model.tubelet_size_enc :]
            for x in expert_obses:
                x["visual"] = x["visual"][-agent.model.tubelet_size_enc :]
        agent.set_goal(prepare_obs(agent.cfg.task_specification.obs, goal_obs))

        if cfg.logging.optional_plots and cfg.task_specification.goal_source in ["dset", "expert"]:
            expert_video_path = str(vis_work_dir / f"expert_video")
            self.ep_expert_frameslist = [x["visual"] for x in expert_obses]
            make_video(self.ep_expert_frameslist, 30, expert_video_path, obs_concat_channels=env.obs_concat_channels)
            if cfg.meta.get("analyze_distances_expert", True):
                analyze_distances(
                    agent,
                    expert_obses,
                    goal_obs,
                    str(dist_work_dir / f"expert"),
                    objective=agent.objective,
                )
        # # why reset agent env ? This makes sur the beginning state of agent is the same
        # as the expert, since we froze rand vec to the same value in get_goal_state()

        self.prev_losses = []
        self.prev_elite_losses_mean = []
        self.prev_elite_losses_std = []
        self.prev_pred_frames_over_iterations = []
        self.predicted_best_encs_over_iterations = []

        def agent_actor(obs, steps_left, plan_vis_path=None):
            act = agent.act(
                prepare_obs(agent.cfg.task_specification.obs, obs),
                steps_left=steps_left,
            )
            self.prev_losses.append(agent._prev_losses)
            self.prev_elite_losses_mean.append(agent._prev_elite_losses_mean)
            self.prev_elite_losses_std.append(agent._prev_elite_losses_std)
            self.prev_pred_frames_over_iterations.append(agent._prev_pred_frames_over_iterations)
            self.predicted_best_encs_over_iterations.append(agent._predicted_best_encs_over_iterations)
            return act

        episode_obses, ep_reward, planned_actions, infos, success, state_dist = self.unroll_agent(
            env,
            init_obs,
            info,
            agent_actor,
            preprocessor=agent.preprocessor,
        )
        if "droid" in cfg.task_specification.task:
            success_dist = 0.0
            # first 6 action dims are summable in time and should lead to same delta whatever the path taken if no obstacles
            total_delta_traj = torch.abs(
                planned_actions[0].sum(0) - self.expert_actions[0][: planned_actions[0].shape[0]].sum(0)
            )
            # sum over 6 action dims
            end_distance_xyz = total_delta_traj[:3].sum().item()
            end_distance_orientation = total_delta_traj[3:6].sum().item()
            end_distance_closure = total_delta_traj[6:].sum().item()
            end_distance = end_distance_xyz + end_distance_orientation + end_distance_closure
        else:
            if cfg.logging.optional_plots and cfg.task_specification.num_frames > agent.model.tubelet_size_enc:
                # Keep the minimal last state
                for x in episode_obses:
                    x["visual"] = x["visual"][-agent.model.tubelet_size_enc :]
            agent_goal_video_path = str(vis_work_dir / f"video_agent_goal_{'succ' if success else 'fail'}")
            frames_list = [x["visual"] for x in episode_obses]
            make_video(frames_list, 30, agent_goal_video_path, obs_concat_channels=env.obs_concat_channels)
            make_video_pdf(
                frames_list[:: self.cfg.frameskip],
                agent_goal_video_path + ".pdf",
                obs_concat_channels=env.obs_concat_channels,
            )

            coord_diffs, _repr_diffs = analyze_distances(
                agent,
                episode_obses,
                goal_obs,
                str(dist_work_dir / "agent"),
                objective=agent.objective,
            )
            success_dist = float(coord_diffs[-1] < 0.05)
            end_distance = coord_diffs[-1]
            end_distance_xyz, end_distance_orientation, end_distance_closure = -1.0, -1.0, -1.0
        if cfg.logging.optional_plots:
            plot_losses(
                self.prev_losses,
                self.prev_elite_losses_mean,
                self.prev_elite_losses_std,
                work_dir=work_dir,
                frameskip=cfg.frameskip,
                num_act_stepped=agent.planner.num_act_stepped,
            )
        if (
            cfg.task_specification.goal_source != "random_state"
            and self.expert_actions is not None
            and cfg.logging.optional_plots
        ):
            plot_actions_comparison(
                planned_actions,  # List[T / n_opt_steps, act_dim] of length n_opt_steps
                self.expert_actions,  # [1, T, act_dim] tensor
                work_dir=work_dir,
                frameskip=cfg.frameskip,
                num_act_stepped=agent.planner.num_act_stepped,
            )
            if cfg.meta.get("compare_unrolled_plan_expert", True):
                expert_embeddings = (
                    agent.model.encode(
                        prepare_obs(agent.cfg.task_specification.obs, torch.stack(expert_obses).to(agent.device)),
                        act=False,
                    )
                    .detach()
                    .cpu()
                )
                total_lpips, total_emb_l2 = compare_unrolled_plan_expert(
                    agent,
                    self.predicted_best_encs_over_iterations,
                    expert_embeddings,
                    self.prev_pred_frames_over_iterations,
                    torch.stack(self.ep_expert_frameslist).squeeze(1),
                )
            else:
                total_lpips, total_emb_l2 = -1.0, -1.0
        else:
            total_lpips, total_emb_l2 = -1.0, -1.0

        return (
            expert_success,
            success,
            ep_reward,
            success_dist,
            end_distance,
            end_distance_xyz,
            end_distance_orientation,
            end_distance_closure,
            state_dist,
            total_lpips,
            total_emb_l2,
        )
