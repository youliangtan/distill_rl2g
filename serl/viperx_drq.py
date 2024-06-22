#!/usr/bin/env python3

# TODO action plan
# 1. faster training speed with 4090
# 2. mixin for reward classifier?
# 3. non mask for termination of reward 2.
# 4. Implement IBRL  https://arxiv.org/pdf/2311.02198


import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import cv2
import os

import pickle as pkl
import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.networks.reward_classifier import load_classifier_func

from agentlace.trainer import TrainerServer, TrainerClient, TrainerConfig
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import make_wandb_logger, make_replay_buffer, make_bc_agent
from serl_launcher.agents.continuous.bc import BCAgent

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

# explicitly set the memory allocation to avoid OOM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

##############################################################################

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "moma-serl-task1", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string(
    "exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 4, "critic to actor update ratio.")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 200000,
                     "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 0, # TODO switch back
                     "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 200,
                     "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 20,
                     "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer(
    "eval_n_trajs", 5, "Number of trajectories for evaluation.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("manipulator_ip", "localhost",
                    "IP address of the manipulator.")

# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_integer("checkpoint_period", 10000, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_boolean("show_img", False, "Show images on the actor")

# debug mode will disable wandb logging
flags.DEFINE_boolean("debug", False, "Debug mode.")

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")
flags.DEFINE_string("preload_online_rlds_path", None, "Path to preload online RLDS data.")

flags.DEFINE_integer(
    "load_checkpoint_step", 0, "Provide a trained policy ckpt at this step"
)
flags.DEFINE_string("task", "task1", "task name")
flags.DEFINE_string(
    "reward_classifier_ckpt_path", None, "Path to reward classifier ckpt."
)

# bc agent
flags.DEFINE_string("bc_chkpt_path", None, "Path to BC ckpt.")
flags.DEFINE_integer("bc_chkpt_step", 0, "Step to load BC ckpt.")

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)
np.set_printoptions(precision=3, suppress=True)


def print_green(x): return print("\033[92m {}\033[00m".format(x))
def print_yellow(x): return print("\033[93m {}\033[00m".format(x))


EEF_DISPLACEMENT = 0.02

##############################################################################

def make_drq_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="small",
    discount=0.98,
):
    """
    NOTE: modified version of the make_drq_agent function in the serl_launcher
    """
    agent = DrQAgent.create_drq(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": jax.nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": jax.nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=10,
        critic_subsample_size=1, # NOTE YL: 1 default 2.
    )
    return agent


def make_trainer_config():
    return TrainerConfig(
        port_number=5288,
        broadcast_port=5289,
        request_types=["send-stats"],
        experimental_pipeline_port=5290, # experimental ds update
    )

def get_image_keys(env):
    return [key for key in env.observation_space.keys() if key != "state"]

##############################################################################


def env_reward_wrapper(
    env, task_config, sampling_rng: jax.random.PRNGKey
) -> gym.Env:
    """
    Similar to env = EnvWrapper(env, ...)
    """
    reward_wrapper_kwargs = task_config.reward_wrapper_kwargs
    if FLAGS.reward_classifier_ckpt_path:
        sampling_rng, key = jax.random.split(sampling_rng)
        classifier_func = load_classifier_func(
            key=key,
            sample=env.observation_space.sample(),
            image_keys=get_image_keys(env),
            checkpoint_path=FLAGS.reward_classifier_ckpt_path,
        )
        reward_wrapper_kwargs["reward_classifier_func"] = classifier_func

    reward_wrapper = task_config.reward_wrapper

    # Wrapper, similar to: env = Wrapper(env)
    return reward_wrapper(env, **reward_wrapper_kwargs)

##############################################################################


def actor(agent: DrQAgent, task_config: dict,
          data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    # NOTE: for experimental IBRL
    if FLAGS.bc_chkpt_path:
        bc_agent: BCAgent = make_bc_agent(
            FLAGS.seed,
            env.observation_space.sample(),
            env.action_space.sample(),
            encoder_type=FLAGS.encoder_type,
            image_keys=get_image_keys(env),
        )
        print_yellow(f"loading bc agent: {FLAGS.bc_chkpt_path} {FLAGS.bc_chkpt_step}")
        ckpt = checkpoints.restore_checkpoint(
            FLAGS.bc_chkpt_path,
            bc_agent.state,
            step=FLAGS.bc_chkpt_step,
        )
        bc_agent = bc_agent.replace(state=ckpt)

    reset_args = dict(
        target_state=task_config.reset_pose,
    )

    env = env_reward_wrapper(env, task_config, sampling_rng)

    ##########################################################
    # Actor Eval mode
    ##########################################################
    if FLAGS.load_checkpoint_step:
        print("we will run checkpoint evaluation with step: ",
              FLAGS.load_checkpoint_step)
        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path,
            agent.state,
            step=FLAGS.load_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        success_counter = 0
        time_list = []
        # implementation for policy evaluation
        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset(**reset_args)
            done = False
            start_time = time.time()
            while not done:
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                )
                actions = np.asarray(jax.device_get(actions))

                obs, reward, done, truncated, info = env.step(actions)

                if FLAGS.show_img:
                    for k in ["image_primary", "image_wrist"]:
                        img = next_obs[k][0]
                        cv2.imshow(k, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    cv2.waitKey(10)

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")
        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # return function when done
    ##########################################################
    # End Actor Eval
    ##########################################################

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        print_green("update actor weights")
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    # training loop
    timer = Timer()
    running_return = 0.0
    curr_step = 0
    obs, _ = env.reset(**reset_args)
    done = False

    @partial(jax.jit)
    def jit_forward_critic(observations, actions, rng):
        return agent.forward_critic(observations, actions, rng, train=False)

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )

                ## experimental ibrl impl
                if FLAGS.bc_chkpt_path:
                    bc_actions = bc_agent.sample_actions(
                        observations=jax.device_put(obs),
                        argmax=True,
                    )
                    # get val from critic
                    drq_q = jit_forward_critic(
                        observations=jax.device_put(obs),
                        actions=actions,
                        rng=sampling_rng,
                        # train=False,
                    )
                    # get val from critic
                    bc_q = jit_forward_critic(
                        observations=jax.device_put(obs),
                        actions=bc_actions,
                        rng=sampling_rng,
                        # train=False,
                    )
                    # print_yellow(f"bc_q: {bc_q}, drq_q: {drq_q}")
                    # bc_q = bc_q.mean(axis=0)
                    # drq_q = drq_q.mean(axis=0)
                    # take max
                    bc_q = bc_q.max(axis=0)
                    drq_q = drq_q.max(axis=0)
                    if bc_q > drq_q:
                        print_yellow(f"using bc actions")
                        actions = bc_actions
                        # clip within 1, -1
                        actions = jnp.clip(actions, -1.0, 1.0)
                    else:
                        print_yellow(f"using drq actions")

                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):
            start_time = time.time()
            next_obs, reward, done, truncated, info = env.step(actions)
            reward = np.asarray(reward, dtype=np.float32)
            print_green(f" - reward!: {reward}, with time: {time.time() - start_time}")
            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done, # TODO maybe to mask
                dones=done or truncated,
            )
            data_store.insert(transition)

            if FLAGS.show_img:
                for k in ["image_primary", "image_wrist"]:
                    img = next_obs[k][0]
                    cv2.imshow(k, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                cv2.waitKey(10)

            obs = next_obs
            if done or truncated:
                print_green(f"episode return: {running_return}")
                client.request("send-stats", {
                    "eps/episode_return": running_return,
                    "eps/episode_length": curr_step,
                })
                running_return = 0.0
                curr_step = 0

                # incoporate random reset box if provided
                if task_config.random_reset_box is not None:
                    box = task_config.random_reset_box
                    reset_args["target_state"][:3] = np.random.uniform(
                        low=box[0], high=box[1]
                    )
                obs, _ = env.reset(**reset_args)

        if step % FLAGS.steps_per_update == 0:
            client.update()

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)

        curr_step += 1


##############################################################################


def learner(rng, agent: DrQAgent,
            replay_buffer, demo_buffer):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    if FLAGS.load_checkpoint_step:
        print_green(
            f"loading policy ckpt with step: {FLAGS.load_checkpoint_step}")
        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path,
            agent.state,
            step=FLAGS.load_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(),
                           request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience if
    # demo_buffer is provided
    if demo_buffer is None:
        single_buffer_batch_size = FLAGS.batch_size
        demo_iterator = None
    else:
        single_buffer_batch_size = FLAGS.batch_size // 2
        demo_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": single_buffer_batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )

    # create replay buffer iterator
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": single_buffer_batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # show replay buffer progress bar during training
    pbar = tqdm.tqdm(
        total=FLAGS.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)

                # we will concatenate the demo data with the online data
                # if demo_buffer is provided
                if demo_iterator is not None:
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_critics(
                    batch,
                )

        with timer.context("train"):
            batch = next(replay_iterator)

            # we will concatenate the demo data with the online data
            # if demo_buffer is provided
            if demo_iterator is not None:
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        # publish the updated network
        if step > 0 and step % (FLAGS.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log(
                {"timer": timer.get_average_times()}, step=update_steps)

        if FLAGS.checkpoint_path and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path,
                agent.state,
                step=update_steps,
                keep=20,
                overwrite=True,
            )

        pbar.update(len(replay_buffer) - pbar.n)  # update replay buffer bar
        update_steps += 1


##############################################################################


def main(_):
    assert FLAGS.batch_size % num_devices == 0
    # get config for the task
    task_config = get_task_config(FLAGS.task)

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    print("Using manipulator gym env")

    if FLAGS.learner:
        print_green("learner mode")
        interface = ManipulatorInterface()  # testing mode
    elif FLAGS.actor:
        print_green("actor mode")
        interface = ActionClientInterface(host=FLAGS.manipulator_ip)
    else:
        raise ValueError("Must be either a learner or an actor")

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
    env = MaxEpsLengthWrapper(env, FLAGS.max_traj_length)
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

    image_keys=get_image_keys(env)
    print(f"image_keys: {image_keys}")
    obs, _ = env.reset(target_state=task_config.reset_pose)

    for key in obs.keys():
        print(f"key: {key}, shape: {obs[key].shape}")
    print("\033[91m {}\033[00m".format("-"*50))

    rng, sampling_rng = jax.random.split(rng)
    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: DrQAgent = jax.device_put(
        jax.jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    ##############################################################################

    if FLAGS.learner:
        sampling_rng = jax.device_put(
            sampling_rng, device=sharding.replicate())
        print_green("replay buffer created")

        # get the reward wrapper
        if task_config.reward_wrapper is None:
            env_rew_func = None
        else:
            env_rew_func = env_reward_wrapper(env, task_config, sampling_rng)

        # NOTE: this assumes that all data are expert demonstrations from a
        # pick up task. TODO: relabel rewards for N step before the end of eps
        total_reward = 0.0
        skip_curr_eps = False
        print_yellow(f" prenorm_action_space low:  {prenorm_action_low}")
        print_yellow(f" prenorm_action_space high: {prenorm_action_high}")

        #########################################################################
        # custom rlpd data transform START
        #########################################################################

        def preload_data_transform(data, metadata):
            nonlocal total_reward, skip_curr_eps
            REWARD_STEPS = 8

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

            # # will only use the data if the language text contains "pick"
            # if "pick" not in metadata["language_text"]:
            #     return None

            # End of Trajectory
            if metadata["step"] == metadata["step_size"] - 1:
                data["dones"] = True  # explicitly set dones to True
                data["masks"] = 0  # explicitly set masks to 0
                print(
                    f"total reward in eps: {total_reward} , with size: {metadata['step_size']}")
                print("-"*50)
                total_reward = 0.0

                if skip_curr_eps:
                    skip_curr_eps = False
                    return None

            # if custom reward function is not provided
            if env_rew_func is None:
                # relabel the reward to 1 when terminated or truncated
                # and 0 otherwise
                # if metadata["step"] == metadata["step_size"] - 1:
                if metadata["step"] > metadata["step_size"] - REWARD_STEPS:
                    data["rewards"] = 1.0
                else:
                    data["rewards"] = 0.0

            else:
                rew, done = env_rew_func.compute_reward(obs)
                data["rewards"] = rew
                # print(f" step: {metadata['step']}, rew: {rew}") # sanity check

                # Custom logic to skip the current episode
                if skip_curr_eps:
                    # print("skipping current step") # sanity check
                    return None
                elif done:
                    data["dones"] = True  # explicitly set dones to True
                    data["masks"] = 0  # explicitly set masks to 0
                    skip_curr_eps = True
                    total_reward += rew # for debugging
                    # print("MARKING DONE") # sanity check
                    return None
                total_reward += rew # for debugging

            return data

        #########################################################################
        # custom rlpd data transform END
        #########################################################################

        # online buffer
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            rlds_logger_path=FLAGS.log_rlds_path, # option to add more path with interleave
            type="memory_efficient_replay_buffer",
            image_keys=image_keys,
            preload_rlds_path=FLAGS.preload_online_rlds_path, # TODO: dont log again
            preload_data_transform=preload_data_transform,
        )

        if FLAGS.preload_rlds_path:
            print_green("loading data from RLDS")
            demo_buffer = make_replay_buffer(
                env,
                capacity=FLAGS.replay_buffer_capacity,
                type="memory_efficient_replay_buffer",
                image_keys=image_keys,
                preload_rlds_path=FLAGS.preload_rlds_path,
                preload_data_transform=preload_data_transform,
            )
            print_green(f"demo_buffer size: {len(demo_buffer)}")
        else:
            demo_buffer = None

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
        )

    ##############################################################################

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(2000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        actor(agent, task_config, data_store, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
