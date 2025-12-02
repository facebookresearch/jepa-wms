# Simu_env_planning eval

## Overview
In this evaluation, we perform goal-conditioned trajectory optimization. We optimize over the action space to minimize the below cost, denoted $C$ (where the $\hat{z}$ sequence is defined recursively):
$$
C(a, s_0, s_g) = \sum_{t=0}^H \| E_{\theta}(s_g) - P_{\theta}(\hat{z}_t, a_t) \|_2, \\
\hat{z}_0 = E_{\theta}(s_0), \quad \hat{z}_{t+1} = P_{\theta}(\hat{z}_t, a_t), \quad t=0,\dots, H.
$$

We define an **evaluation episode** as a pair $(s_0, s_g)$, the *task definition* along with the *plan* outputted by our agent and planning procedure, and whether this leads to $s_g$, i.e. **Success** or not, i.e. **Failure**.

### Goal sources
We have 4 potential goal sources supported:
- **dset**: the initial and goal state are sampled from the validation set. Used for Push-T, Robocasa and DROID. The **random_actions** variant steps actions sampled from a Gaussian from the initial state sampled in the trajectory.
- **expert**: for Metaworld, starting from a randomly sampled initial state, we play the expert policy provided with the environment to get an expert trajectory and goal state.
- **random_state**: For Maze and Wall, the 2D position of the initial and goal state are sampled with a function built in the simulator.
For tasks where `goal_source != random_state`, we can thus store an *expert trajectory*, leading from the initial to the goal state.


### Episode Metrics
This evaluation outputs the following metrics:
- **Success Rate**: the percentage of success across the `cfg.meta_eval_episodes` evaluation episodes.
-  Distance to goal at end of episode (`ep_end_dist`)
-  For Metaworld, Maze, Push-T and Wall: the average cumulative reward over the episodes (`reward`), as defined in the simulator.
-  For Metaworld, log whether the arm has reached the target location, regardless of the objects involved in the success definition (`success_dist`)
- For tasks where `goal_source != random_state`, unroll the world model on the expert actions, compute between the visual states hereby produced and those produced by unrolling the plan outputted by the agent:
  - the L2 distance between respective visual embeddings (`total_emb_l2`)
  - the LPIPS between respective visual decodings (`total_lpips`)
-  For DROID offline planning only, we also track:
   -  `end_distance_xyz`: distance to goal end-effector orientation
   -  `end_distance_orientation`: distance to goal end-effector orientation
   -  `end_distance_closure`: distance to goal gripper closure

### Optional episode analysis plots

The eval optionally produces the following plots for each eval episode if `cfg.logging.optional_plots`, using the functions in `evals/simu_env_planning/planning/episode_plot_utils.py`. We display an example figure of a Robocasa evaluation episode for each of the below plots.
- `state.pdf`: the initial and goal images $s_0$ and $s_g$ used for the *task definition*.
- `video_agent_goal_{success}.gif`: GIF of the episode, after stepping the planned actions in the environment.
- `expert_video.gif`: if `goal_source in ["dset", "expert"]` GIF of the expert trajectory stepped in the environment.

| State | Expert video | Agent success video |
|------------|-------------|---------------|
| <img src="assets/rcasa_state.png" width="320" /> | <img src="assets/expert_video.gif" width="180" />| <img src="assets/video_agent_goal_succ.gif" width="180" /> |

- `losses.pdf`: For each timestep of the episode where the agent plans, track the planning cost $C$ throughout iterations of the planning optimizer.
- `action_comparison.pdf`: comparison, for each action dimension, of the value outputted by the agent and the one of the expert, throughout episode timesteps.

Planning costs | Actions comparison to expert actions|
|-------------|---------------|
<img src="assets/rcasa_losses.png"  width="450" />| <img src="assets/droid_action_comp.png"  width="400" /> |

- `agent_rep_distance_visual.pdf` and `expert_rep_distance_visual.pdf`: visual embedding $L_2$ distance throughout timesteps stepped by the agent in the env and the goal visual embedding.
- `agent_rep_distance_proprio.pdf` and `expert_rep_distance_proprio.pdf`: proprioceptive embedding $L_2$ distance throughout timesteps stepped by the agent in the env and the goal proprioceptive embedding.
- `agent_objectives` and `expert_objectives`: planning objective $C$ of successive timesteps stepped by the agent in the env.
- `agent_distances.pdf` and `expert_distances`: cartesian distance of the agent (end-effector for Metaworld, DROID, and Robocasa) to the goal.

| | Expert | Agent |
|---|-------------|---------------|
Cartesian Distance |<img src="assets/rcasa_expert_distances.png"  width="300" />| <img src="assets/rcasa_agent_distances.png"  width="300" /> |
Representation Distance |<img src="assets/rcasa_expert_repdist_vis.png"  width="300" />| <img src="assets/rcasa_agent_repdist_vis.png"  width="300" /> |

Moreover, if `cfg.planner.decode_each_iteration` we get the following plots:
- `step{k}.gif`, `step{k}_gt.pdf` and `step{k}_last_frames.pdf`: at each episode step where the agent plans, the GIF displays the best plan of each planning optimizer (e.g. CEM) iteration. `step{k}_last_frames.pdf` is the decoding of the unrolling of the actions outputted by the planning optimizer. `step{k}_gt.pdf` is the consequence of stepping this plan in the environment.

<!-- | step0.gif | step0_last_frames | step0_gt  |
|------------|-------------|---------------|
| <img src="assets/step0.gif"  width="200" /> | <img src="assets/rcasa_step0_lastframes.png"  width="250" />| <img src="assets/rcasa_step0_gt.png"  width="250" /> |
| |Stepping of only the first action out of the $H=3$ actions planned, both in the env and in the world model's "imagination"|| -->
<table>
<tr>
<th>step0.gif</th>
<th>step0_last_frames</th>
<th>step0_gt</th>
</tr>
<tr>
<td><img src="assets/step0.gif" width="200" /></td>
<td><img src="assets/rcasa_step0_lastframes.png" width="250" /></td>
<td><img src="assets/rcasa_step0_gt.png" width="250" /></td>
</tr>
<tr>
<td></td>
<td colspan="2">Stepping of only the first action out of the H=3 actions planned, both in the env and in the world model's "imagination".</td>
</tr>
</table>

### Distributed episodes
This eval parallelizes by distributing independent episodes accross GPUs, via the `main_distributed_episodes_eval()` function in `evals/simu_env_planning/eval.py`.

### Planning optimization
It supports various:
- Planners (maintained ones are CEM and NeverGrad) at `evals/simu_env_planning/planning/planning/planner.py`
- Planning objectives at `evals/simu_env_planning/planning/planning/objectives.py`
- Goal sources: expert for Metaworld, random state or dset for Push-T, Wall and Maze. They are managed in the `set_episode()` function of `evals/simu_env_planning/planning/plan_evaluator.py`

### Model wrapper for planning
For this eval to be run, we require a wrapper, the default one being `app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds.EncPredWM`.
In the wrapper, we initiliaze:
- the **world model** (encoder, predictor and optional other modules).

This wrapper requires prior loading, in in `evals/simu_env_planning/eval.py`, of:
- a **data_preprocessor**. It is used to normalize and denormalize the actions and the proprioception if we used normalize_action = True. We should use denormalization of planned actions outputted by the world model so they can be stepped in the associated simulator.
- the **validation set**. It is needed:
  - to determine the action and proprioception input dimensions
  - to determine the mean and std of actions and proprioception in the data proprocesor
  - to provide the data to define the planning task via the init and goal state $s_0$ and $s_g$ if `goal_source == dset`

## Eval Configs
You can find:
- Some full example configs at `configs/evals/simu_env_planning` to evaluate on the Metaworld, Robocasa envs or on offline DROID dataset trajectories.
- Some configs templates for the same environments to fill with your model kwargs at `evals/simu_env_planning/base_configs`
