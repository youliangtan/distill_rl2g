#!/usr/bin/env python3

import numpy as np
import math
from wrappers import (
    FancyRewardClassifierWrapperWithGripper,
    GoToRewardWrapper,
)


##############################################################################

class GoToTarget:
    name = "GoToTarget"
    workspace_boundary = np.array(
        [[0.15, -0.3, -0.15],
         [0.55, 0.3, 0.4]]
    )
    reset_pose = np.array([0.26, 0.0, 0.1, 0.0, math.pi/2, 0.0, 1.0]) # open
    random_reset_box = np.array(
        [[0.2, -0.2, 0.1],
         [0.4, 0.2, 0.]]
    )
    reward_wrapper=GoToRewardWrapper
    reward_wrapper_kwargs=dict(
        gripper_open=False,
        target_pos=np.array([0.3, 0.0, -0.15]),
        dense_reward=False,
    )
    action_dim = 4
    with_gripper=True


##############################################################################

class PickUpSquishy(GoToTarget):
    name = "PickUpSquishy"
    workspace_boundary = np.array(
        # [[0.18, -0.15, -0.15],
        #  [0.4, 0.15, 0.2]]
        [[0.18, -0.14, -0.15], # NOTE: smaller workspace
         [0.4, 0.14, 0.2]]
    )
    # TODO update and use this reset pose
    random_reset_box = None
    reset_pose = np.array([0.26, 0.0, 0.10, 0.0, math.pi/2, 0.0, 1.0]) # open
    reward_wrapper=FancyRewardClassifierWrapperWithGripper
    reward_wrapper_kwargs=dict(
        terminate_on_n_reward=5,
        target_z=-0.14, # same as goal_pose[2]
        target_z_lift=-0.12,  # NOTE: use sparse reward when provided
    )

##############################################################################

class InsertThePlug(GoToTarget):
    name = "InsertThePlug"
    workspace_boundary = np.array(
        [[0.18, -0.15, -0.15],
         [0.4, 0.15, 0.2]]
    )
    reset_pose = np.array([0.26, 0.0, 0.1, 0.0, math.pi/2, 0.0, 0.0]) # keep close
    random_reset_box = np.array(
        [[0.18, -0.15, 0.1],
         [0.4, 0.15, 0.]]
    )
    reward_wrapper=GoToRewardWrapper
    reward_wrapper_kwargs=dict(
        with_gripper=False,
        target_pos=np.array([0.3, 0.0, -0.15]),
        dense_reward=False,
    )
    action_dim = 3 # no gripper
    with_gripper=False


##############################################################################

class SweepTheSweets:
    # TODO: work on this
    pass


##############################################################################

class PlaceTheSponge(GoToTarget):
    workspace_boundary = np.array(
        [[0.15, -0.3, -0.15],
         [0.55, 0.3, 0.4]]
    )
    reset_pose = np.array([0.26, 0.0, 0.1, 0.0, math.pi/2, 0.0, 1.0]) # open
    random_reset_box = None


##############################################################################

def get_task_config(task_name: str):
    if task_name == "task0":
        return GoToTarget()
    elif task_name == "task1":
        return PickUpSquishy()
    elif task_name == "task2":
        return InsertThePlug()
    else:
        raise ValueError(f"Task {task_name} not found in task_configs.py")

##############################################################################

# Testing
if __name__ == "__main__":
    import gym
    task_config = get_task_config("task1")
    print(task_config.workspace_boundary)
    print(task_config.reset_pose)
    print(get_task_config("task2"))

    # pendulum for test
    task_config = get_task_config("task0")
    test_env = gym.make("Pendulum-v1")
    print(task_config.name)
    reward_wrapper = task_config.reward_wrapper
    print(reward_wrapper)
    env = reward_wrapper(test_env, **task_config.reward_wrapper_kwargs)
    print("done")
