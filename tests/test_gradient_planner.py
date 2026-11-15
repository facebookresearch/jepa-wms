# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from evals.simu_env_planning.planning.planning.planner import AdamPlanner, GradientDescentPlanner


class TestGradientPlanner(unittest.TestCase):
    def test_optimizer_trajectory(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z = torch.tensor([0.7, -1.1], device=device)

        def unroll(z, act_suffix):
            return z + act_suffix.cumsum(0)

        def objective(encodings, actions):
            return encodings.square().mean()

        for planner_cls, optimizer_cls, kwargs in (
            (GradientDescentPlanner, torch.optim.SGD, {}),
            (AdamPlanner, torch.optim.Adam, {"betas": (0.5, 0.8), "eps": 0.02}),
        ):
            with self.subTest(planner=planner_cls.__name__):
                planner = planner_cls(
                    unroll,
                    action_dim=2,
                    horizon=3,
                    iterations=4,
                    lr=0.2,
                    sample_type="zero",
                    action_noise=0,
                    max_norms=[0.15],
                    max_norm_dims=[[1]],
                    adam_betas=(0.5, 0.8),
                    adam_eps=0.02,
                )
                planner.device = device
                planner.set_objective(objective)
                result = planner.plan(z, steps_left=2)

                actions = torch.zeros(2, 1, 2, device=device, requires_grad=True)
                optimizer = optimizer_cls([actions], lr=0.2, **kwargs)
                for pred in result.predicted_best_encs_over_iterations:
                    optimizer.zero_grad()
                    objective(unroll(z, actions), actions).backward()
                    optimizer.step()
                    with torch.no_grad():
                        actions[..., 1].clamp_(-0.15, 0.15)
                        torch.testing.assert_close(pred, unroll(z, actions))
                torch.testing.assert_close(result.actions, actions[:, 0])

    def test_only_actions_receive_gradients(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weight = torch.tensor(2.0, device=device, requires_grad=True)
        z = weight.square()
        weight.grad = torch.ones_like(weight)
        planner = AdamPlanner(
            lambda z, act_suffix: z + weight * act_suffix,
            action_dim=1,
            horizon=1,
            iterations=2,
            sample_type="zero",
            action_noise=0,
        )
        planner.device = device
        planner.set_objective(lambda encodings, actions: encodings.square().mean())
        planner.plan(z)
        torch.testing.assert_close(weight.grad, torch.ones_like(weight))
