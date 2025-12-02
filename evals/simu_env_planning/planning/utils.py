import random
from io import BytesIO

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tensordict.tensordict import TensorDict

from src.utils.logging import get_logger

log = get_logger(__name__)

FIGSIZE_BASE = (4.0, 3.0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_td(obs, info):
    return TensorDict(
        {
            "visual": obs,
            "proprio": info["proprio"],
        },
        batch_size=[],
    )


def prepare_obs(obs_type, td):
    if obs_type == "rgb_state":
        return td
    elif obs_type == "state":
        return td["proprio"]
    elif obs_type == "rgb":
        return td["visual"]
    else:
        raise ValueError(f"Unknown observation type: {obs_type}")


###### Video, GIF and pdf saving utils ##########
def make_video(images, fps, output_path, obs_concat_channels=True):
    make_video_mp4(images, fps, output_path + ".mp4", obs_concat_channels=obs_concat_channels)
    make_video_gif(images, fps, output_path + ".gif", obs_concat_channels=obs_concat_channels)


def make_video_mp4(images, fps, output_path, obs_concat_channels=True):
    """
    Input:
        images: List[tau C H W] of length T
    """
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    for img in images:
        img = (img[-3:] if obs_concat_channels else img[-1]).numpy() if isinstance(img, torch.Tensor) else img
        img = img.transpose(1, 2, 0)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        # img: H W C
        writer.append_data(img)

    writer.close()
    print(f"Video saved to {output_path}")


def make_video_gif(images, fps, output_path, obs_concat_channels=True):
    writer = imageio.get_writer(output_path, fps=fps, format="GIF", loop=10000)
    for img in images:
        img = (img[-3:] if obs_concat_channels else img[-1]).numpy() if isinstance(img, torch.Tensor) else img
        img = img.transpose(1, 2, 0)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        writer.append_data(img)

    writer.close()
    print(f"GIF saved to {output_path}")


def make_video_pdf(images, output_path, obs_concat_channels=True):
    """
    Save video frames as a PDF with frames arranged horizontally

    Args:
        images: List of images [tau C H W] of length T
        output_path: Path to save the PDF
        obs_concat_channels: Whether the observation has concatenated channels
    """
    # Process images to consistent format
    processed_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            img = (img[-3:] if obs_concat_channels else img[-1]).detach().cpu().numpy()

        # Convert to HWC format
        img = img.transpose(1, 2, 0)
        processed_images.append(img)

    # Create a single horizontal image by concatenating all frames
    concat_img = np.concatenate(processed_images, axis=1)

    # Create a figure with exact dimensions (no margins)
    fig = plt.figure(figsize=(concat_img.shape[1] / 300, concat_img.shape[0] / 300), dpi=300)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Display the concatenated image
    ax.imshow(concat_img)

    # Save without any padding
    plt.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"PDF saved to {output_path}")


def save_init_goal_frame(init_obs, goal_obs, vis_work_dir=None, concat_channels=False):
    obs_img = init_obs["visual"]
    obs_img = obs_img[-3:] if concat_channels else obs_img[-1]
    goal_state_img = goal_obs["visual"]
    goal_state_img = goal_state_img[:3] if concat_channels else goal_state_img[0]
    plt.figure(dpi=300, figsize=FIGSIZE_BASE)
    plt.subplot(1, 2, 1)
    plt.imshow(obs_img.permute(1, 2, 0))
    plt.title("Init state")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(goal_state_img.permute(1, 2, 0))
    plt.title("Goal state")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(vis_work_dir / f"state.pdf", bbox_inches="tight")
    plt.close()


def save_decoded_frames(pred_frames_over_iterations, costs, plan_vis_path, overlay=True):
    """
    costs: Tensor[float] of length iterations
    pred_frames_over_iterations: List[(T, H, W, C)] of length iterations
    """
    if pred_frames_over_iterations is not None and plan_vis_path is not None:
        frames = []
        global_min_cost = costs.min()
        global_max_cost = costs.max()
        # Pre-calculate the normalized positions for all costs
        all_normalized_costs = []
        for i in range(len(costs)):
            # For each iteration, normalize all costs seen so far
            current_costs = costs[: i + 1]
            if len(current_costs) > 1:
                # Normalize using global min/max for consistent scaling
                normalized = (current_costs - global_min_cost) / (global_max_cost - global_min_cost + 1e-10)
                all_normalized_costs.append(normalized)
            else:
                all_normalized_costs.append(np.array([0.5]))  # Default for single value

        for i, pred_frames in enumerate(pred_frames_over_iterations):
            # pred_frames.shape: (T, H, W, C)
            if overlay:
                overlay_frames = []
                for frame_idx, frame in enumerate(pred_frames):
                    # Create a copy of the frame to draw on
                    frame_with_overlay = frame.copy()

                    # Get frame dimensions
                    h, w = frame.shape[0], frame.shape[1]

                    # Get normalized costs for this iteration
                    current_costs = costs[: i + 1]
                    if len(current_costs) > 1:
                        normalized_costs = all_normalized_costs[i]
                        # Map to pixel space (top is low cost, bottom is high cost)
                        y_positions = (1 - normalized_costs) * (h - 20) + 10

                        # Draw the white curve with anti-aliasing
                        for j in range(len(y_positions) - 1):
                            x1 = int(j * w / len(costs))  # Fixed x-scale based on total iterations
                            x2 = int((j + 1) * w / len(costs))
                            y1 = int(y_positions[j])
                            y2 = int(y_positions[j + 1])
                            cv2.line(frame_with_overlay, (x1, y1), (x2, y2), (200, 200, 200), 2, cv2.LINE_AA)

                        # Add cost values at key points
                        # Show initial cost value
                        cost_text = f"{current_costs[0].item():.2f}"
                        cv2.putText(
                            frame_with_overlay,
                            cost_text,
                            (10, int(y_positions[0])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (200, 200, 200),
                            1,
                        )

                        # Show final cost value
                        cost_text = f"{current_costs[-1].item():.2f}"
                        cv2.putText(
                            frame_with_overlay,
                            cost_text,
                            (int(x1) - 50, int(y_positions[-1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (200, 200, 200),
                            1,
                        )
                        # Add min/max values
                        cv2.putText(
                            frame_with_overlay,
                            f"Min: {global_min_cost:.2f}",
                            (w - 120, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (200, 200, 200),
                            1,
                        )
                        cv2.putText(
                            frame_with_overlay,
                            f"Max: {global_max_cost:.2f}",
                            (w - 120, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (200, 200, 200),
                            1,
                        )

                    # Add text showing the iteration number
                    cv2.putText(
                        frame_with_overlay, f"Iter {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2
                    )

                    overlay_frames.append(frame_with_overlay)
                frames.extend(overlay_frames)
            else:
                plt.clf()
                plt.figure(figsize=(10, 10))
                plt.plot(costs[: i + 1])
                plt.title(f"Iteration {i}")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.xlim(0, len(costs))
                plt.ylim(min(costs), max(costs))
                buf = BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                img = Image.open(buf)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (256, 256))

                combined_frames = []
                for frame in pred_frames:
                    frame = cv2.resize(frame, (256, 256))
                    combined_frame = np.concatenate((img, frame), axis=1)
                    combined_frames.append(combined_frame)
                frames.extend(combined_frames)
        # Save GIF
        filename = f"{plan_vis_path}.gif"
        duration = 0.1
        imageio.mimsave(filename, frames, duration=duration, loop=0)
        log.info(f"Plan decoding video saved to {plan_vis_path}")

        # Save last iteration frames as PDF
        last_pred_frames = pred_frames_over_iterations[-1]
        pdf_filename = f"{plan_vis_path}_last_frames.pdf"
        n_frames = len(last_pred_frames)
        fig, axes = plt.subplots(1, n_frames, figsize=(5 * n_frames, 5))
        if n_frames == 1:  # Handle case with single frame
            axes = [axes]

        for ax, frame in zip(axes, last_pred_frames):
            ax.imshow(frame)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
        plt.close()
        log.info(f"Last iteration frames saved to {pdf_filename}")


##########################################################


######### Distributed results aggregation utils ##########
def aggregate_results(cfg, all_results):
    combined_results = {}
    task_episode_counts = {task: 0 for task in cfg.tasks}

    for task in cfg.tasks:
        combined_results[f"episode_reward+{task}"] = 0
        combined_results[f"episode_success+{task}"] = 0
        combined_results[f"ep_expert_succ+{task}"] = 0
        combined_results[f"ep_succ_dist+{task}"] = 0
        combined_results[f"ep_end_dist+{task}"] = 0
        combined_results[f"ep_end_dist_xyz+{task}"] = 0
        combined_results[f"ep_end_dist_orientation+{task}"] = 0
        combined_results[f"ep_end_dist_closure+{task}"] = 0
        combined_results[f"ep_time+{task}"] = 0
        combined_results[f"ep_state_dist+{task}"] = 0
        combined_results[f"ep_total_lpips+{task}"] = 0
        combined_results[f"ep_total_emb_l2+{task}"] = 0

    for results in all_results:
        for key, value in results.items():
            task = key.split("+")[1]
            combined_results[key] += value[0]  # Sum up results from all GPUs
            task_episode_counts[task] += value[1]  # Sum up episode counts

    num_metrics_per_task = len(combined_results) // len(cfg.tasks)

    for task in task_episode_counts:
        task_episode_counts[task] /= num_metrics_per_task
        assert (
            task_episode_counts[task] == cfg.meta.eval_episodes
        ), f"Task {task} has {task_episode_counts[task]} episodes, expected {cfg.meta.eval_episodes}"

    for key in combined_results.keys():
        task = key.split("+")[1]
        if task_episode_counts[task] > 0:
            combined_results[key] /= cfg.meta.eval_episodes

    return combined_results


def compute_task_distribution(cfg):
    """
    Each rank is given a (start_episode, end_episode) that are global indices among the
    total_episodes = cfg.meta.eval_episodes * len(cfg.tasks) episodes to perform.
    Returns:
    - task_indices: List[int]
    - episodes_per_task: List[List[int]] with length=len(task_indices): each sublist contains the global
    episode indices for a specific task that the current rank is responsible for evaluating.
    episodes_per_task[i] is the episodes indices list for task at position i in task_indices.
    """
    if cfg.distributed.distribute_multitask_eval:
        total_episodes = cfg.meta.eval_episodes * len(cfg.tasks)
        episodes_per_rank = total_episodes // cfg.world_size
        extra_episodes = total_episodes % cfg.world_size
        max_episodes_per_rank = episodes_per_rank + (1 if extra_episodes > 0 else 0)

        # GPUs with a rank less than extra_episodes have
        start_episode = cfg.rank * episodes_per_rank + min(cfg.rank, extra_episodes)
        # GPUs with a rank less than extra_episodes will handle one additional episode
        end_episode = start_episode + episodes_per_rank + (1 if cfg.rank < extra_episodes else 0)
        print(f"Rank {cfg.rank}: {start_episode=}, {end_episode=}")
        task_indices = []
        episodes_per_task = []

        # 0 <= start_episode, end_episode <= total_episodes
        # rasterized order starting by task, then eps per task
        for episode_idx in range(start_episode, end_episode):
            task_idx = episode_idx // cfg.meta.eval_episodes
            episode_num = episode_idx % cfg.meta.eval_episodes
            # if not added ep for this task in task_indices
            if not task_indices or task_indices[-1] != task_idx:
                task_indices.append(task_idx)
                episodes_per_task.append([])
            episodes_per_task[-1].append(episode_num)

        # For each GPU to perform same number of eval episodes, add dummy ones
        current_total_episodes = end_episode - start_episode
        if current_total_episodes < max_episodes_per_rank:
            dup_eps_needed = max_episodes_per_rank - current_total_episodes
            for _ in range(dup_eps_needed):
                # Duplicate the last episode of the last task
                if episodes_per_task:
                    episodes_per_task[-1].append(episodes_per_task[-1][-1])

        return task_indices, episodes_per_task
    else:
        task_indices = list(range(len(cfg.tasks)))
        episodes_per_task = [list(range(cfg.meta.eval_episodes)) for _ in task_indices]
        return task_indices, episodes_per_task


#################################
