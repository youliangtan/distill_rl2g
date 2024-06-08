"""
Script to record goals and failed transitions for the bin relocation task.

Usage:
    python record_transitions.py --transitions_needed 400

add `--record_failed_only` to only record failed transitions
"""

import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import cv2
import time

from manipulator_gym.manipulator_env import ManipulatorEnv, StateEncoding
from manipulator_gym.interfaces.interface_service import ActionClientInterface
from manipulator_gym.interfaces.base_interface import ManipulatorInterface
from manipulator_gym.utils.gym_wrappers import (
    ConvertState2Proprio,
    ResizeObsImageWrapper,
)
from serl_launcher.wrappers.chunking import ChunkingWrapper

from vr_data_collection import OculusReaderListener, TeleopDataCollector

from task_configs import get_task_config
import argparse

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--transitions_needed", type=int, default=400, help="number of transitions to collect",)
    arg_parser.add_argument("--label", type=str, default="", help="label for the file")
    arg_parser.add_argument("--robot_ip", help="IP address of the robot server.", default="localhost", type=str)
    arg_parser.add_argument("--oculus_server_ip", help="oculus server ip", default="localhost", type=str)
    arg_parser.add_argument("--img_size", help="image size", default=128, type=int)
    arg_parser.add_argument("--rotate_oculus", help="rotate oculus", default=False, type=bool)
    arg_parser.add_argument("--task", help="task name", default="task1", type=str)

    args = arg_parser.parse_args()
    ep_done = False

    task_config = get_task_config(args.task)

    env = ManipulatorEnv(
        workspace_boundary=task_config.workspace_boundary,
        manipulator_interface=ActionClientInterface(host=args.robot_ip),
        # manipulator_interface=ManipulatorInterface(), # for testing
        state_encoding=StateEncoding.POS_EULER,
        use_wrist_cam=True,
    )
    env = ResizeObsImageWrapper(
        env,
        resize_size={
            "image_primary": (args.img_size, args.img_size),
            "image_wrist": (args.img_size, args.img_size)
        }
    )
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    # this is to turn 180 degrees so that the oculus is facing the person
    yaw_correction = np.pi if args.rotate_oculus else 0.0

    reader = OculusReaderListener(
        host=args.oculus_server_ip,
        apply_yaw_correction=yaw_correction,
    )
    data_collector = TeleopDataCollector(
        oculus_reader=reader,
        env=env,
        shared_controller=False,
        resize_img=False
    )

    obs, _ = env.reset(target_state=task_config.reset_pose)
    recorded_transitions = []
    _pbar = tqdm(total=args.transitions_needed, desc="transitions")

    def check_all_done():
        return (
            len(recorded_transitions) >= args.transitions_needed
        )

    # Loop until we have enough transitions
    while not check_all_done():
        actions, ep_done, _ = data_collector.get_action()
        if actions is None:
            print("paused...")
            time.sleep(0.1)
            continue

        next_obs, rew, done, _, info = env.step(actions)

        cv2.imshow("prim_image", cv2.cvtColor(
            next_obs["image_primary"][0], cv2.COLOR_BGR2RGB))
        cv2.imshow("wrist_image", cv2.cvtColor(
            next_obs["image_wrist"][0], cv2.COLOR_BGR2RGB))
        cv2.waitKey(10)

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )

        if (len(recorded_transitions) < args.transitions_needed):
            recorded_transitions.append(transition)
            _pbar.update(1)
        obs = next_obs
        print("EP DONE", ep_done)

        if ep_done:
            obs, _ = env.reset(target_state=task_config.reset_pose)
        time.sleep(0.1)

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # save transitions
    file_name = f"serl_transitions_{args.label}_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(recorded_transitions, f)
        print(f"saved {len(recorded_transitions)} transitions to {file_name}")

    print("done")
