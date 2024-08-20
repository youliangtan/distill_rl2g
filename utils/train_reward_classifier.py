from typing import Callable, Optional
import pickle as pkl
import jax
from jax import numpy as jnp
import flax
import flax.linen as nn
from flax.training import checkpoints
import optax
from tqdm import tqdm
import gym
import os
from absl import app, flags
import numpy as np
import cv2

from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop, batched_color_transform
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.networks.reward_classifier import create_classifier


# Set above env export to prevent OOM errors from memory preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("positive_demo_paths", None,
                          "paths to positive demos")
flags.DEFINE_multi_string("negative_demo_paths", None,
                          "paths to negative demos")
flags.DEFINE_string("classifier_ckpt_path", ".",
                    "Path to classifier checkpoint")
flags.DEFINE_integer("batch_size", 256, "Batch size for training")
flags.DEFINE_integer("num_epochs", 100, "Number of epochs for training")


def _populate_data_store(
    data_store: MemoryEfficientReplayBufferDataStore,
    demos_path: str,
    transform: Optional[Callable] = None,
    resize_image_keys = None,
):
    """
    Utility function to populate demonstrations data into data_store.
    :Args:
        data_store (MemoryEfficientReplayBufferDataStore): Data store to populate.
        demos_path (str): Path to the demonstrations.
        transform (Optional[Callable]): Transform function to apply to the data. Defaults to None.
    :return data_store
    """
    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                # resize to 128x128 if resize_size is provided
                if resize_image_keys is not None:
                    for key in resize_image_keys:
                        transition["observations"][key] = cv2.resize(
                            transition["observations"][key], (128, 128))
                        transition["next_observations"][key] = cv2.resize(
                            transition["next_observations"][key], (128, 128))

                # HACK method to apply transform to the observations and actions
                if transform is not None:
                    for key in ["observations", "next_observations", "actions"]:
                        transition[key] = transform(transition[key])

                data_store.insert(transition)
    print(f"Populated data store with {len(data_store)} transitions.")
    return data_store


def get_gym_space(data_sample) -> gym.spaces.Space:
    """
    Get the data type as Gym space of a provided data sample.
    This function currently only supports common types like Box and Dict type.

    :Args:  data_sample (Any): The data sample to be converted.
    :Returns:   gym.spaces.Space
    """
    # Case for numerical data (numpy arrays, lists of numbers, etc.)
    if isinstance(data_sample, (np.ndarray, list, tuple)):
        # Ensure it's a numpy array to get shape and dtype
        data_sample = np.array(data_sample)
        if np.issubdtype(data_sample.dtype, np.integer):
            low = np.iinfo(data_sample.dtype).min
            high = np.iinfo(data_sample.dtype).max
        elif np.issubdtype(data_sample.dtype, np.inexact):
            low = float("-inf")
            high = float("inf")
        else:
            raise ValueError()
        return gym.spaces.Box(low=low, high=high,
                              shape=data_sample.shape, dtype=data_sample.dtype)
    # Case for dictionary data
    elif isinstance(data_sample, dict):
        # Recursively convert each item in the dictionary
        return gym.spaces.Dict({key: get_gym_space(value)
                                for key, value in data_sample.items()})
    elif isinstance(data_sample, (int, float, str)):
        return gym.spaces.Discrete(1)
    else:
        raise TypeError("Unsupported data type for Gym spaces conversion.")


def add_chunking_dim(data):
    """
    Add a chunking dimension to the data.
    """
    if isinstance(data, dict):
        return {k: add_chunking_dim(v) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        return np.expand_dims(data, axis=0)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


##############################################################################

def train_reward_classifier(observation_space, action_space, is_chunked):
    """
    User can provide custom observation space to be used as the
    input to the classifier. This function is used to train a reward
    classifier using the provided positive and negative demonstrations.

    NOTE: some datas are chunked, some are not. The replay buffer requires the 
    data to be chunked before storing it. Thus, the argument is_chunked is used

    NOTE: this function is duplicated and used in both
    async_bin_relocation_fwbw_drq and async_cable_route_drq examples
    """
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)

    print("positive_demo_paths: ", FLAGS.positive_demo_paths)
    print("negative_demo_paths: ", FLAGS.negative_demo_paths)

    image_keys = [k for k in observation_space.keys() if "state" not in k]
    # check if the observation or action is chunked

    pos_buffer = MemoryEfficientReplayBufferDataStore(
        observation_space,
        action_space,
        capacity=10000,
        image_keys=image_keys,
    )

    # we will apply the chunking dimension to the data if it is not chunked
    transition_transform = None if is_chunked else add_chunking_dim
    pos_buffer = _populate_data_store(
        pos_buffer, FLAGS.positive_demo_paths, transition_transform, resize_image_keys=image_keys)

    neg_buffer = MemoryEfficientReplayBufferDataStore(
        observation_space,
        action_space,
        capacity=10000,
        image_keys=image_keys,
    )
    neg_buffer = _populate_data_store(
        neg_buffer, FLAGS.negative_demo_paths, transition_transform, resize_image_keys=image_keys)

    print(f"failed buffer size: {len(neg_buffer)}")
    print(f"success buffer size: {len(pos_buffer)}")
    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    rng, key = jax.random.split(rng)
    classifier = create_classifier(key, sample["observations"], image_keys)

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )

        # # NOTE: the original image is in uint8, and the color_transform function
        # # requires float32, thus we need to convert the image to float32 first
        # # then convert it back to uint8 after the color transformation
        observations = observations.copy(
            add_or_replace={
                pixel_key: jnp.array(
                    observations[pixel_key], dtype=jnp.float32)
                / 255.0,
            }
        )
        observations = observations.copy(
            add_or_replace={
                pixel_key: batched_color_transform(
                    observations[pixel_key],
                    rng,
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.1,
                    apply_prob=1.0,
                    to_grayscale_prob=0.0,  # don't convert to grayscale
                    color_jitter_prob=0.5,
                    shuffle=False,  # wont shuffle the color channels
                    num_batch_dims=2,  # 2 images observations
                ),
            }
        )
        observations = observations.copy(
            add_or_replace={
                pixel_key: jnp.array(
                    observations[pixel_key] * 255.0, dtype=jnp.uint8
                ),
            }
        )
        return observations

    # Define the training step
    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["data"], rngs={"dropout": key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn(
            {"params": state.params}, batch["data"], train=False, rngs={"dropout": key}
        )
        # print("logits: ", nn.sigmoid(logits))
        train_accuracy = jnp.mean(
            (nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    # Training Loop
    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        # Merge and create labels
        sample = concat_batches(
            pos_sample["next_observations"], neg_sample["next_observations"], axis=0
        )
        rng, key = jax.random.split(rng)
        sample = data_augmentation_fn(key, sample)
        labels = jnp.concatenate(
            [
                jnp.ones((FLAGS.batch_size // 2, 1)),
                jnp.zeros((FLAGS.batch_size // 2, 1)),
            ],
            axis=0,
        )
        batch = {"data": sample, "labels": labels}

        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(
            classifier, batch, key)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

    # this is used to save the without the orbax checkpointing
    flax.config.update('flax_use_orbax_checkpointing', False)
    checkpoints.save_checkpoint(
        FLAGS.classifier_ckpt_path,
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )


def main(_):
    # read the first transition from the positive demo
    with open(FLAGS.positive_demo_paths[0], 'rb') as f:
        pos_first_transition = pkl.load(f)[0]

    observation_sample = pos_first_transition["observations"]

    image_keys = [k for k in observation_sample.keys() if "state" not in k]
    # resize all imgages to 128x128
    for key in image_keys:
        observation_sample[key] = cv2.resize(
            observation_sample[key], (128, 128))

    action_sample = pos_first_transition["actions"]

    is_chunked = False if len(action_sample.shape) == 1 else True

    # we will add chunking dimension to the data if it is not chunked
    if not is_chunked:
        observation_sample = add_chunking_dim(observation_sample)
        action_sample = add_chunking_dim(action_sample)

    observation_space = get_gym_space(observation_sample)
    action_space = get_gym_space(action_sample)
    print("observation_space: ", observation_space)
    print("action_space: ", action_space)

    train_reward_classifier(observation_space, action_space, is_chunked)


if __name__ == "__main__":
    app.run(main)
