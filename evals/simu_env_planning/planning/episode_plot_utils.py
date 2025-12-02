from typing import Callable, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tensordict import TensorDict

matplotlib.use("Agg")

from app.plan_common.models.lpips import LPIPS
from evals.simu_env_planning.planning.utils import prepare_obs

FIGSIZE_BASE = (4.0, 3.0)
from src.utils.logging import get_logger

log = get_logger(__name__)


def plot_losses(
    losses,
    elite_losses_mean,
    elite_losses_std,
    work_dir,
    frameskip=1,
    num_act_stepped=1,
):
    """
    Input:
        losses: List[Tensor, size= (n_opt_steps, n_losses)]
        elite_losses_mean: List[Tensor, size= (n_opt_steps, n_losses)]
        elite_losses_std: List[Tensor, size= (n_opt_steps, n_losses)]
    For now, n_losses = 1.
    """
    losses = torch.stack(losses, dim=0).detach().cpu().numpy()
    elite_losses_mean = torch.stack(elite_losses_mean, dim=0).detach().cpu().numpy()
    elite_losses_std = torch.stack(elite_losses_std, dim=0).detach().cpu().numpy()
    n_timesteps, n_opt_steps, n_losses = losses.shape
    sns.set_theme()
    for i in range(n_losses):
        total_plots = min(16, n_timesteps)
        rows = 1
        cols = int(np.ceil(total_plots / rows))
        fig_width = FIGSIZE_BASE[0] * cols
        fig_height = FIGSIZE_BASE[1] * rows
        plt.figure(figsize=(fig_width, fig_height), dpi=300)
        steps = np.linspace(0, n_timesteps - 1, total_plots, dtype=int)
        for j, step in enumerate(steps):
            ax = plt.subplot(rows, cols, j + 1)
            if n_opt_steps > 1:
                sns.lineplot(data=losses[step, :, i])
                sns.lineplot(data=elite_losses_mean[step, :, i])
                ax.fill_between(
                    range(n_opt_steps),
                    elite_losses_mean[step, :, i] - elite_losses_std[step, :, i],
                    elite_losses_mean[step, :, i] + elite_losses_std[step, :, i],
                    alpha=0.3,
                )
            else:
                ax.bar(0, losses[step, 0, i])  # Plot a bar chart if only one opt step
                ax.bar(0, elite_losses_mean[step, 0, i])
                ax.errorbar(0, elite_losses_mean[step, 0, i], yerr=elite_losses_std[step, 0, i], fmt="none", capsize=5)
            ax.set_title(f"Episode step {step * frameskip * num_act_stepped}")
            ax.set_xlabel("Opt step")
            if j == 0:
                ax.set_ylabel("Traj cost")
            ax.tick_params(axis="both")
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(work_dir / f"losses_{i}.pdf", bbox_inches="tight")
        plt.close()


def plot_actions_comparison(
    agent_actions,
    expert_actions,
    work_dir,
    frameskip=1,
    num_act_stepped=1,
):
    """
    Plot the norm of planned actions from agent compared to expert actions.

    Args:
        agent_actions: List[Tensor] of agent actions from plan_evaluator.unroll_agent()
                    Shape: (tau, A), length O
        expert_actions: Tensor of expert actions from plan_evaluator.unroll_expert()
                    Shape: (1, T, A)
        T <= tau * O
    """
    n_opt_steps = len(agent_actions)
    expert_timesteps, action_dim = expert_actions.shape[1], expert_actions.shape[2]
    # Make a list out of expert actions to match agent actions format
    expert_act = []
    s = 0
    for i in range(n_opt_steps):
        expert_act.append(expert_actions[0, s : s + agent_actions[i].shape[0]].detach().cpu().numpy())
        s += agent_actions[i].shape[0]

    sns.set_theme()
    total_plots = min(16, n_opt_steps)
    rows = 1
    cols = int(np.ceil(total_plots / rows))
    legend_width = 3.0
    fig_width = FIGSIZE_BASE[0] * cols + legend_width
    legend_fraction = 1.0 / cols
    # legend_fraction = legend_width / fig_width
    fig_height = FIGSIZE_BASE[1] * rows
    plt.figure(figsize=(fig_width, fig_height), dpi=300)

    colors = plt.cm.tab10(np.linspace(0, 1, action_dim))

    steps = np.linspace(0, n_opt_steps - 1, total_plots, dtype=int)

    legend_handles = []
    legend_labels = []

    for j, step in enumerate(steps):
        ax = plt.subplot(rows, cols, j + 1)

        agent_shape = agent_actions[step].shape[0]
        expert_shape = expert_act[step].shape[0]
        if agent_shape == 1 and expert_shape == 1:
            # Bar chart for single timestep
            width = 0.35
            x = np.arange(action_dim)
            agent_vals = agent_actions[step][0].detach().cpu().numpy()
            expert_vals = expert_act[step][0]
            agent_bars = ax.bar(x - width / 2, agent_vals, width, color=colors, label="Agent")
            expert_bars = ax.bar(
                x + width / 2, expert_vals, width, color=colors, alpha=0.5, label="Expert", hatch="//"
            )
            if j == 0:
                for dim in range(action_dim):
                    legend_handles.append(agent_bars[dim])
                    legend_labels.append(f"Agt {dim}")
                    legend_handles.append(expert_bars[dim])
                    legend_labels.append(f"Exp {dim}")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Dim {dim}" for dim in range(action_dim)])
        else:
            for dim in range(action_dim):
                agent_line = ax.plot(
                    range(agent_actions[step].shape[0]),
                    agent_actions[step][:, dim].detach().cpu().numpy(),
                    linestyle="-",
                    color=colors[dim],
                    marker="o",
                    markersize=2,
                )[0]

                expert_line = ax.plot(
                    range(expert_act[step].shape[0]),
                    expert_act[step][:, dim],
                    linestyle="--",
                    color=colors[dim],
                    marker="o",
                    markersize=2,
                )[0]

                # Only collect handles and labels from the first subplot for the legend
                if j == 0:
                    legend_handles.append(agent_line)
                    legend_handles.append(expert_line)
                    legend_labels.append(f"Agt {dim}")
                    legend_labels.append(f"Exp {dim}")
            ax.set_xlabel("Time")
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_title(f"Episode step {step * frameskip * num_act_stepped}")
        if j == 0:
            ax.set_ylabel("Act value")
        ax.tick_params(axis="both")

    # Add a single legend to the first subplot
    # Create separate handles for agent/expert linestyles
    style_handles = []
    style_labels = []
    # Solid line for agent
    style_handles.append(matplotlib.lines.Line2D([], [], color="black", linestyle="-"))
    style_labels.append("Agent")
    # Dashed line for expert
    style_handles.append(matplotlib.lines.Line2D([], [], color="black", linestyle="--"))
    style_labels.append("Expert")

    # Create separate handles for dimension colors
    dim_handles = []
    dim_labels = []
    for dim in range(action_dim):
        dim_handles.append(matplotlib.lines.Line2D([], [], color=colors[dim]))
        dim_labels.append(f"Dim {dim}")

    left_pos = 1  # Position from left edge
    bottom_pos = 0.95  # Position from bottom

    fig = plt.gcf()

    # Style legend (left)
    first_legend = fig.legend(
        style_handles, style_labels, loc="upper left", bbox_to_anchor=(left_pos, bottom_pos), title=None
    )
    fig.add_artist(first_legend)

    # Get the width of the first legend for positioning the second one
    first_legend_width = first_legend.get_window_extent().width / fig.dpi / fig_width
    second_legend_pos = left_pos + first_legend_width

    # Dimension legend (right)
    second_legend = fig.legend(
        dim_handles, dim_labels, loc="upper left", bbox_to_anchor=(second_legend_pos, bottom_pos), title=None
    )

    plt.tight_layout()
    plt.savefig(work_dir / "action_comparison.pdf", bbox_inches="tight")
    plt.close()
    return expert_act


def analyze_distances(
    agent,
    obses,
    goal_obs,
    plot_prefix,
    objective: Optional[Callable] = None,
):
    """
    Input:
        obses: List[
            TensorDict(
                proprio: [tau Prop]
                visual: [tau, c, h, w]
            )
        ] of length env.max_episode_steps + 1
    """
    coords = torch.stack([x["proprio"] for x in obses])  # B tau Prop
    distances = (
        torch.norm(coords[..., -1, :3] - goal_obs["proprio"][-1, :3].unsqueeze(0), dim=-1).detach().cpu().numpy()
    )
    if agent.cfg.logging.optional_plots:
        sns.set_theme()
        FIGSIZE = (4.0, 3.0)
        plt.figure(figsize=FIGSIZE, dpi=300)
        sns.lineplot(data=distances)
        plt.xlabel("Timesteps")
        plt.ylabel("Distance to goal")
        plt.tight_layout()
        plt.savefig(plot_prefix + "_distances.pdf", bbox_inches="tight")
        plt.close()

    # encode all states
    # Careful: this use of torch.stack requires obses and goal_obs to be TensorDicts or Tensors
    all_states = torch.stack(obses + [goal_obs]).to(agent.device)
    # The encoder takes batch of single states, of dim [len_ep, tubelet_size_enc, C, H, W] so no temporal dependency
    # We could encode the obses with temporal dependency but rather see them as a batch of images
    all_encs = agent.model.encode(prepare_obs(agent.cfg.task_specification.obs, all_states), act=True)
    # compute latent L2 distance of trajectory of the agent to the goal embedding
    diffs = compute_embed_differences(all_encs).detach().cpu().numpy()
    if agent.cfg.logging.optional_plots:
        for key, value in diffs.items():
            plt.figure(figsize=FIGSIZE, dpi=300)
            sns.lineplot(data=value)
            plt.xlabel("Timesteps")
            plt.ylabel("Rep distance to goal")
            plt.tight_layout()
            plt.savefig(plot_prefix + f"_rep_distance_{key}.pdf", bbox_inches="tight")
            plt.close()
        if objective:
            if isinstance(all_encs, TensorDict):
                all_encs_excluded = TensorDict({key: value[:-1] for key, value in all_encs.items()})
            else:
                all_encs_excluded = all_encs[:-1]
            all_objectives = objective(all_encs_excluded, None, keepdims=True).detach().cpu()
            plt.figure(figsize=FIGSIZE, dpi=300)
            sns.lineplot(data=all_objectives.squeeze(1))
            plt.xlabel("Timesteps")
            plt.ylabel("Objective values")
            plt.tight_layout()
            plt.savefig(plot_prefix + "_objectives.pdf", bbox_inches="tight")
            plt.close()

    return distances, diffs


def compute_embed_differences(all_encs):
    """
    Input: all_encs:
        visual: [T, 1, ..., D]
        proprio: [T, 1, ..., P]
    Output:

    """
    if isinstance(all_encs, TensorDict):
        return TensorDict(
            {
                "visual": (all_encs["visual"][:-1] - all_encs["visual"][-1:])
                .pow(2)
                .mean(dim=tuple(range(1, all_encs["visual"].ndim))),
                "proprio": (all_encs["proprio"][:-1] - all_encs["proprio"][-1:])
                .pow(2)
                .mean(dim=tuple(range(1, all_encs["proprio"].ndim))),
            }
        )
    else:
        return TensorDict({"visual": (all_encs[:-1] - all_encs[-1:]).pow(2).mean(dim=tuple(range(1, all_encs.ndim)))})


def compare_unrolled_plan_expert(
    agent,
    unrolled_embeddings,
    expert_embeddings,
    pred_frames_over_iterations=None,
    expert_frames=None,
):
    """
    Compute LPIPS between decoded plan (latest planner itr) and expert frames for each for o in [1,O],
        then averaged.
        TODO: and optionally L2 for proprio decodings
    Compute L2 distance between unrolled plan and expert embeddings, for both the visual and
        optionally the proprio embeddings.
    Input:
        unrolled_embeddings: List[List[TensorDict or Tensor[tau, 1, 1, H, W, D]] of len planner.iterations]
            of length O the number of calls to agent.act()
        expert_embeddings: TensorDict or Tensor[T, 1, 1, H, W, D] of length T the episode length
        pred_frames_over_iterations: List[List[ndarray[tau, H, W, C]] of len planner.iterations] of length O
        expert_frames: Tensor[T, H, W, C]] of len T the episode length
    T <= tau * O
    """
    lpips = LPIPS().eval().to(agent.device)  # expects [0,1]
    s, total_lpips, total_emb_l2 = 1, 0.0, 0.0
    if agent.planner.decode_each_iteration:
        for opt_step, pred_frames_list in enumerate(pred_frames_over_iterations):
            curr_pred_frames = (
                torch.tensor(pred_frames_list[-1], dtype=torch.float32)[s:].permute(0, 3, 1, 2) / 255.0
            )  # T H W C -> T C H W
            curr_expert_frames = (
                expert_frames[s : s + len(curr_pred_frames)].to(dtype=torch.float32) / 255.0
            )  # T C H W
            s += len(curr_pred_frames)
            total_lpips += lpips(curr_pred_frames.to(agent.device), curr_expert_frames.to(agent.device)).mean().item()

    s = 1  # Reset counter for embeddings
    if unrolled_embeddings is not None and expert_embeddings is not None:
        for opt_step, embed_list in enumerate(unrolled_embeddings):
            if isinstance(embed_list[-1], TensorDict):
                curr_embed = embed_list[-1]["visual"][s:]  # Use the last iteration, skip first frame
                curr_expert_embed = expert_embeddings["visual"][s : s + len(curr_embed)]
            else:
                curr_embed = embed_list[-1][s:]
                curr_expert_embed = expert_embeddings[s : s + len(curr_embed)]
            s += len(curr_embed)
            total_emb_l2 += torch.nn.functional.mse_loss(
                curr_embed.to(agent.device), curr_expert_embed.to(agent.device)
            ).item()

    return total_lpips, total_emb_l2
