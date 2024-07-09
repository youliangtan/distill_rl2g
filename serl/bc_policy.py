#!/usr/bin/env python3

from tqdm import tqdm
from absl import app, flags
from flax.training import checkpoints
import jax
from jax import numpy as jnp
import numpy as np
import time
import cv2

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.utils.launcher import (
    make_bc_agent,
    make_wandb_logger,
    make_replay_buffer,
)
from serl_launcher.networks.reward_classifier import load_classifier_func

# local imports
from task_configs import get_task_config
from wrappers import (
    ResizeEnvAction,
    MaxEpsLengthWrapper,
)
from manipulator_gym.utils.gym_wrappers import ResizeObsImageWrapper, ClipActionBoxBoundary
from manipulator_gym.interfaces.base_interface import ManipulatorInterface
from manipulator_gym.interfaces.interface_service import ActionClientInterface
from manipulator_gym.manipulator_env import ManipulatorEnv, StateEncoding


FLAGS = flags.FLAGS

flags.DEFINE_string("env", "distill_rl2g_bc", "Name of environment.")
flags.DEFINE_string("agent", "bc", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", True, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")

flags.DEFINE_integer("max_steps", 20000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 10000, "Replay buffer capacity.")
# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")
flags.DEFINE_integer(
    "eval_checkpoint_step", 0, "evaluate the policy from ckpt at this step"
)
flags.DEFINE_integer("eval_n_trajs", 100, "Number of trajectories for evaluation.")
flags.DEFINE_boolean("debug", False, "Debug mode.")  # debug mode will disable wandb logging

# custom to distill_rl2g
flags.DEFINE_string("task", "task1", "task name")
flags.DEFINE_string("manipulator_ip", "localhost",
                    "IP address of the manipulator.")
flags.DEFINE_boolean("show_img", False, "Show images")

def print_green(x): return print("\033[92m {}\033[00m".format(x))
def print_yellow(x): return print("\033[93m {}\033[00m".format(x))

EEF_DISPLACEMENT = 0.02


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)
np.set_printoptions(precision=3, suppress=True)


def main(_):
    assert FLAGS.batch_size % num_devices == 0
    rng = jax.random.PRNGKey(FLAGS.seed)

    # get config for the task
    task_config = get_task_config(FLAGS.task)

    if not FLAGS.eval_checkpoint_step:
        print_green("training mode")
        interface = ManipulatorInterface()  # testing mode
    else:
        print_green("eval mode")
        interface = ActionClientInterface(host=FLAGS.manipulator_ip)

    env = ManipulatorEnv(
        manipulator_interface=interface,
        state_encoding=StateEncoding.POS_EULER,
        use_wrist_cam=True,
        eef_displacement=EEF_DISPLACEMENT,
    )
    env = ClipActionBoxBoundary(
        env,
        workspace_boundary=task_config.workspace_boundary,
        out_of_boundary_penalty=0.01,
    )
    env = ResizeObsImageWrapper(
        env, resize_size={"image_primary": (
            128, 128), "image_wrist": (128, 128)}
    )
    # env = MaxEpsLengthWrapper(env, FLAGS.max_traj_length)
    env = ResizeEnvAction(
        env,
        task_config.action_dim,
        task_config.with_gripper,
        eef_displacement=EEF_DISPLACEMENT,
    )
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    prenorm_action_low = env.action_space.low
    prenorm_action_high = env.action_space.high

    env = gym.wrappers.RescaleAction(env, -1., 1.)
    env = RecordEpisodeStatistics(env)

    print_yellow(f"observation_space: {env.observation_space}")
    print_yellow(f"action_space: {env.action_space}")

    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    rng, sampling_rng = jax.random.split(rng)
    agent: BCAgent = make_bc_agent(
        FLAGS.seed,
        env.observation_space.sample(),
        env.action_space.sample(),
        encoder_type=FLAGS.encoder_type,
        image_keys=image_keys,
        image_augmentation=("random_crop", "color_transform"),
    )

    #########################################################################
    # custom rlpd data transform START
    #########################################################################
    def preload_data_transform(data, metadata):
        obs = data["observations"]
        action = data['actions']
        next_obs = data["next_observations"]

        # NOTE: we will normalize the action if it is the 
        # raw rlds data.
        is_original_rlds = True if len(obs["state"].shape) == 1 else False

        # convert data img shape from (h, w, c) to (History, h, w, c)
        # and resize to 128x128
        for key in image_keys:
            if len(obs[key].shape) == 3:
                obs[key] = cv2.resize(obs[key], (128, 128))
                obs[key] = np.expand_dims(obs[key], axis=0)

            if len(next_obs[key].shape) == 3:
                next_obs[key] = cv2.resize(next_obs[key], (128, 128))
                next_obs[key] = np.expand_dims(next_obs[key], axis=0)

        # convert data state shape from (N,) to (1, N)
        if len(obs["state"].shape) == 1:
            obs["state"] = np.expand_dims(obs["state"], axis=0)
        if len(next_obs["state"].shape) == 1:
            next_obs["state"] = np.expand_dims(next_obs["state"], axis=0)

        # NOTE: resize action dimension according to the action_dim in task_config
        if is_original_rlds:
            mod_action = np.zeros(task_config.action_dim, dtype=np.float32)
            if task_config.with_gripper:
                mod_action[:task_config.action_dim -
                        1] = action[:task_config.action_dim-1]
                mod_action[-1] = action[-1]
            else:
                mod_action[:task_config.action_dim] = action[:task_config.action_dim]
            action = mod_action
        assert len(action) == task_config.action_dim, "action dim not equal to task_config action dim"

        # rescale actions to -1 to 1 if is original rlds data
        if is_original_rlds:
            action = (action - prenorm_action_low) / \
                (prenorm_action_high - prenorm_action_low) * 2.0 - 1.0

        # ensure all actions are within -1 and 1
        assert np.all(action >= -1.0) and np.all(action <=
                1.0), "action not within -1 and 1"

        data["observations"] = obs
        data['actions'] = action
        data["next_observations"] = next_obs
        
        ## NOTE: since this is BC, we will ignore all rewards done etc.
        return data

    #########################################################################
    # custom rlpd data transform END
    #########################################################################


    if not FLAGS.eval_checkpoint_step:
        """
        Training Mode
        """
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        # load demos and populate to current replay buffer
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            type="memory_efficient_replay_buffer",
            image_keys=image_keys,
            preload_rlds_path=FLAGS.preload_rlds_path,
            preload_data_transform=preload_data_transform,
        )
        print_green(f"replay_buffer size: {len(replay_buffer)}")

        replay_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": FLAGS.batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )

        wandb_logger = make_wandb_logger(
            project="serl_bc",
            description=FLAGS.exp_name or FLAGS.env,
            debug=FLAGS.debug,
        )

        for step in tqdm(range(FLAGS.max_steps)):
            batch = next(replay_iterator)
            agent, info = agent.update(batch)

            # log to wandb every 100 steps
            if (step + 1) % 100 == 0:
                wandb_logger.log(info, step=step)

            if (step + 1) % 2000 == 0 and FLAGS.save_model:
                checkpoints.save_checkpoint(
                    FLAGS.checkpoint_path,
                    agent.state,
                    step=step + 1,
                    keep=10,
                    overwrite=True,
                )

            # Run evaluate the agent and log to wandb every 200 steps
            if (step + 1) % 200 == 0:
                wandb_logger.log(agent.get_debug_metrics(batch), step=step)

    else:
        """
        Evaluation Mode
        """
        from pynput import keyboard

        is_failure = False
        is_success = False
        is_paused = False

        def esc_on_press(key):
            nonlocal is_failure, is_success, is_paused
            if key == keyboard.Key.esc:
                is_failure = True
            elif key == keyboard.Key.space and not is_success:
                is_success = True
            elif key == keyboard.Key.shift:
                is_paused = not is_paused

        keyboard_listener = keyboard.Listener(on_press=esc_on_press)
        keyboard_listener.start()

        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        reset_args = dict(
            target_state=task_config.reset_pose,
        )

        success_counter = 0
        time_list = []

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset(**reset_args)
            done = False
            is_failure = False
            is_success = False
            start_time = time.time()
            while not done:
                
                if is_paused:
                    time.sleep(0.1)
                    continue

                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                )
                actions = np.asarray(jax.device_get(actions))

                print_green(f"actions: {actions}")
                # clip actions within -1 and 1
                actions = np.clip(actions, -1.0, 1.0)
                obs, reward, done, truncated, info = env.step(actions)

                if FLAGS.show_img:
                    for k in ["image_primary", "image_wrist"]:
                        img = obs[k][0]
                        cv2.imshow(k, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    cv2.waitKey(10)

                if is_failure:
                    done = True
                    print("terminated by user")

                if is_success:
                    reward = 1
                    done = True
                    print("success, reset now")

                if done:
                    if not is_failure:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

            # wandb_logger.log(info, step=episode)

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")


if __name__ == "__main__":
    app.run(main)
