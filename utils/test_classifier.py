import pickle
import cv2
import numpy as np
from train_reward_classifier import add_chunking_dim
import argparse
from serl_launcher.networks.reward_classifier import load_classifier_func
import jax


def _get_full_obs(): return {
    "image_primary": np.zeros((128, 128, 3)),
    "image_wrist": np.zeros((128, 128, 3)),
}


def sigmoid(x): return 1 / (1 + np.exp(-x))


def display_transitions(pickle_file, reward_classifier_ckpt_path):
    # Load recorded transitions from the pickle file
    with open(pickle_file, 'rb') as f:
        recorded_transitions = pickle.load(f)

    print("loading ckpt: ", reward_classifier_ckpt_path)
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    classifier_func = load_classifier_func(
        key=key,
        sample=_get_full_obs(),
        image_keys=["image_primary", "image_wrist"],
        checkpoint_path=reward_classifier_ckpt_path,
    )

    positive = 0
    negative = 0

    for transition in recorded_transitions:
        primary_img = transition['next_observations']['image_primary']
        wrist_img = transition['next_observations']['image_wrist']

        # # Convert RGB to BGR for OpenCV
        # primary_img = cv2.cvtColor(primary_img, cv2.COLOR_RGB2BGR)
        # wrist_img = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)

        # Display the images
        cv2.imshow("Primary Image", primary_img)
        if wrist_img is not None:
            cv2.imshow("Wrist Image", wrist_img)

        # Wait for a key press and close the windows
        cv2.waitKey(10) & 0xFF

        # Check if the action is chunked (1, N) or (N,)
        is_chunked = False if len(transition["actions"].shape) == 1 else True

        # we will chunk the observations if they are not previously chunked
        if is_chunked:
            obs = transition['next_observations']
        else:
            obs = add_chunking_dim(transition['next_observations'])

        logit = classifier_func(obs).item()
        reward = sigmoid(logit)

        if reward > 0.5:
            positive += 1
        else:
            negative += 1

    print(f" => positive: {positive}, negative: {negative}")
    print(f"Positive Ratio: {positive / (positive + negative)}")
    print(f"Negative Ratio: {negative / (positive + negative)}")

    cv2.destroyAllWindows()


##############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_file", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()

    pickle_file = args.pickle_file
    checkpoint_path = args.checkpoint_path
    display_transitions(pickle_file, checkpoint_path)

    # Example:
    # python test_classifier.py --pickle_file record-positive1.pkl --checkpoint_path checkpoint_20
