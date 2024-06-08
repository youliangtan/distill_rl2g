import gym
from tqdm import tqdm
import numpy as np

# new manipulator gym env
from manipulator_gym.manipulator_env import ManipulatorEnv, StateEncoding
from manipulator_gym.interfaces.interface_service import ActionClientInterface
from manipulator_gym.interfaces.base_interface import ManipulatorInterface
from manipulator_gym.utils.gym_wrappers import ResizeObsImageWrapper

from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from wrappers import BinaryRewardClassifierWrapper

from task_configs import get_task_config

import jax
from absl import app, flags
import time

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "reward_classifier_ckpt_path", None, "Path to reward classifier ckpt."
)
flags.DEFINE_string("robot_ip", "localhost", "IP address of the robot server.")
flags.DEFINE_string("oculus_server_ip", None, "oculus server ip")
flags.DEFINE_string("task", "task1", "task name")

##############################################################################

print_yellow = lambda x: print("\033[93m {}\033[00m".format(x))
print_green = lambda x: print("\033[92m {}\033[00m".format(x))


def main(_):
    config = get_task_config(FLAGS.task)

    env = ManipulatorEnv(
        workspace_boundary=config.workspace_boundary,
        manipulator_interface=ActionClientInterface(host=FLAGS.robot_ip),
        # manipulator_interface=ManipulatorInterface(), # for testing
        state_encoding=StateEncoding.POS_EULER,
        use_wrist_cam=True,
        eef_displacement=0.015
    )
    env = ResizeObsImageWrapper(
        env, resize_size={"image_primary": (128, 128), "image_wrist": (128, 128)}
    )
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    image_keys = [k for k in env.observation_space.keys() if "state" not in k]

    # use teleop if provided
    if FLAGS.oculus_server_ip:
        from vr_data_collection import OculusReaderListener, TeleopDataCollector
        reader = OculusReaderListener(
            host=FLAGS.oculus_server_ip,
            # apply_yaw_correction=yaw_correction,
        )
        data_collector = TeleopDataCollector(
            oculus_reader=reader,
            env=env,
            shared_controller=False,
            resize_img=False
        )

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    classifier_func = load_classifier_func(
        key=key,
        sample=env.observation_space.sample(),
        image_keys=image_keys,
        checkpoint_path=FLAGS.reward_classifier_ckpt_path,
    )
    env = BinaryRewardClassifierWrapper(
        env,
        classifier_func,
    )

    obs, _ = env.reset(target_state=config.reset_pose)

    for i in tqdm(range(1000)):
        
        if FLAGS.oculus_server_ip:
            actions, ep_done, _ = data_collector.get_action()
            if actions is None:
                print("paused...")
                time.sleep(0.1)
                continue
        else:
            ep_done = False # from oculus reader
            actions = np.zeros((7,))

        obs, rew, done, truncated, info = env.step(action=actions)

        if rew >= 0.001:
            print_green(f"Reward: {rew}")
        else:
            print_yellow(f"Reward: {rew}")

        if done or ep_done:
            obs, _ = env.reset(target_state=config.reset_pose)

##############################################################################

if __name__ == "__main__":
    app.run(main)
