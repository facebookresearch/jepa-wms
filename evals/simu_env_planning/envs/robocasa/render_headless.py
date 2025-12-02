import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import robocasa.macros as macros
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils import transform_utils as T
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":

    config = {
        "env_name": "PnPCounterToCab",
        "robots": "TMR_ROBOT",
        "controller_configs": load_composite_controller_config(robot="TMR_ROBOT"),
        "translucent_robot": False,
    }

    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        renderer="mujoco",
        render_camera="robot0_robotview_2",
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        camera_heights=300,
        camera_widths=480,
        camera_names=["robot0_robotview_2"],
        camera_depths=True,
    )

    aux = env.reset()
    env.sim.forward()
    env.robots[0].print_action_info()

    # Target setup
    tar_pos = aux["obj_pos"].copy()
    tar_pos[2] += 0.25
    start_pos = aux["robot0_right_eef_pos"].copy()
    start_pos[2] += 0.25
    fixed_quat = aux["robot0_right_eef_quat_site"]  # Constant orientation

    pose_C = tar_pos

    gripper_state = np.array(
        [
            0.10999999940395355,
            1.190000033378601,
            1.1444294333457947,
            0.50000001192092896,
            0.15000000596046448,
            1.190000033378601,
            1.1700000047683716,
            -0.0019986582919955254,
            0.012606220319867134,
            1.190000033378601,
            0.891339099407196,
            0.5501810997724533,
            1.3960000276565552,
            0.6330000162124634,
            0.6200000047683716,
            0.51999998688697815,
        ]
    )

    env.sim.forward()

    eef_pos_history = np.zeros((600, 3))
    obtained_eef_pos_history = np.zeros((600, 3))
    imgs = []

    for step in range(600):
        eef_pos = pose_C
        eef_rpy = R.from_quat(fixed_quat).as_euler("xyz")
        print(eef_pos)
        action_dict = {
            "right": np.concatenate([eef_pos, eef_rpy]),
            "right_gripper": gripper_state,
        }

        action = env.robots[0].create_action_vector(action_dict)
        aux = env.step(action)

        imgs.append(env.sim.render(300, 300))
        imgs.append(aux[0]["robot0_robotview_2_image"])

    # Save video
    video_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (imgs[0].shape[1], imgs[0].shape[0]))
    for img in imgs:
        out.write(img[::-1, :, ::-1])
    out.release()
