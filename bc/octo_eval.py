"""
This is a script modified from
https://github.com/octo-models/octo/blob/main/examples/03_eval_finetuned.py
"""

from absl import app, logging
import gym
import jax
import numpy as np
import wandb
from typing import Dict, Tuple
import cv2
import tensorflow_datasets as tfds
from manipulator_gym.manipulator_env import ManipulatorEnv, StateEncoding
from manipulator_gym.interfaces.interface_service import ActionClientInterface
from manipulator_gym.interfaces.base_interface import ManipulatorInterface
from manipulator_gym.utils.gym_wrappers import (
    ConvertState2Proprio,
    ResizeObsImageWrapper,
)
import sys
from oxe_envlogger.envlogger import OXEEnvLogger
import random
from datetime import datetime
import argparse
import pickle

import base64
import requests

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import (
    HistoryWrapper,
    RHCWrapper,
    UnnormalizeActionProprio,
    TemporalEnsembleWrapper,
)


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def task_success_classifier(obs: Dict[str, np.ndarray]) -> bool:
    """
    This uses the chatgpt model to determine if the task is successful or not
    Args:
        obs: The observation from the environment
    Returns:
        bool: True if the task is successful, False otherwise
    """
    # OpenAI API Key
    gpt_api_key = "sk-proj-C8aNfrfDIy9ciesN0FvNT3BlbkFJjQe2NiTIPTItc4HMKjew"

    # Path to your image
    print("using chatgpt to query")
    p_img_to_check = obs["image_primary"][0]
    p_img_to_check = cv2.cvtColor(p_img_to_check, cv2.COLOR_BGR2RGB)
    cv2.imwrite("primary_grasp_detection_image.jpg", p_img_to_check)

    w_img_to_check = obs["image_wrist"][0]
    w_img_to_check = cv2.cvtColor(w_img_to_check, cv2.COLOR_BGR2RGB)
    cv2.imwrite("wrist_grasp_detection_image.jpg", w_img_to_check)

    p_image_path = "primary_grasp_detection_image.jpg"
    w_image_path = "wrist_grasp_detection_image.jpg"
    # Getting the base64 string
    base64_image_prim = encode_image(p_image_path)
    base64_image_wrist = encode_image(w_image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {gpt_api_key}",
    }

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I'm providing the third person and wrist camera images of the robot arm. Is the robot arm grabbing the coke can? Please respond with true or false, one word and all lower case.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_prim}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_wrist}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    votes = []

    for i in range(5):
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        res_json = response.json()
        print(res_json)
        label = res_json["choices"][0]["message"]["content"]
        if "true" in label: # TODO (YL), should have a more robust way of checkin this
            votes.append(1)
        else:
            votes.append(0)

    print("chat gpt votes:", votes)
    final_vote = max(set(votes), key=votes.count)
    return final_vote


class OctoEval:
    def __init__(self, env, checkpoint, ip):
        logging.info("Loading finetuned model...")
        if checkpoint:
            self.model = OctoModel.load_pretrained(checkpoint)
        else:
            self.model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")

        self.env = env
        # self.env = env
        self.env = ConvertState2Proprio(self.env)
        self.env = ResizeObsImageWrapper(
            self.env,
            resize_size={"image_primary": (256, 256), "image_wrist": (128, 128)},
        )

        # # add wrappers for history and "receding horizon control", i.e. action chunking
        self.env = HistoryWrapper(self.env, horizon=2)
        # env = HistoryWrapper(env, horizon=1)
        self.env = TemporalEnsembleWrapper(self.env, 4)
        # env = RHCWrapper(env, exec_horizon=4)

        # NOTE: we are using bridge_dataset's statistics for default normalization
        # wrap env to handle action/proprio normalization -- match normalization type to the one used during finetuning
        self.env = UnnormalizeActionProprio(
            self.env, self.model.dataset_statistics, normalization_type="normal"
        )

    def perform_action(self, text_cond, obs):
        language_instruction = [text_cond]
        task = self.model.create_tasks(texts=language_instruction)
        # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
        actions = self.model.sample_actions(
            jax.tree_map(lambda x: x[None], obs), task, rng=jax.random.PRNGKey(0)
        )
        actions = actions[0]
        print("performing action: ", actions)
        # step env -- info contains full "chunk" of observations for logging
        # obs only contains observation for final step of chunk
        obs, reward, done, trunc, info = self.env.step(actions)
        return obs, reward, done, trunc, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        help="Path to Octo checkpoint directory.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--ip", help="IP address of the robot server.", default="localhost", type=str
    )
    parser.add_argument(
        "--text_cond",
        help="Language prompt for the task.",
        default="put the banana on the plate",
        type=str,
    )
    parser.add_argument(
        "--ds_name",
        help="Folder to store octo eval rollouts",
        default="octo_eval",
        type=str,
    )
    parser.add_argument("--no_reset_pose", action="store_true")
    parser.add_argument(
        "--show_img",
        help="Whether to visualize the images or not.",
        default=False,
        type=bool,
    )
    args = parser.parse_args()


    recording_data = False

    # load finetuned model
    logging.info("Loading finetuned model...")
    print(args.checkpoint_path)
    if not args.checkpoint_path:
        model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
    else:
        model = OctoModel.load_pretrained(args.checkpoint_path)

    # make gym environment
    ##################################################################################################################
    # environment needs to implement standard gym interface + return observations of the following form:
    #   obs = {
    #     "image_0": ...
    #     "image_1": ...
    #   }
    # it should also implement an env.get_task() function that returns a task dict with goal and/or language instruct.
    #   task = {
    #     "language_instruction": "some string"
    #     "goal": {
    #       "image_0": ...
    #       "image_1": ...
    #     }
    #   }
    ##################################################################################################################
    env = ManipulatorEnv(
        manipulator_interface=ActionClientInterface(host=args.ip),
        # manipulator_interface=ManipulatorInterface(), # for testing
        state_encoding=StateEncoding.POS_EULER,
        use_wrist_cam=True,
    )

    if recording_data:
        step_metadata_info = {
            "language_text": tfds.features.Text(doc="Language embedding for the episode.")
        }

        env = OXEEnvLogger(
            env,
            dataset_name="octo_eval_env",
            directory=f"{args.ds_name}/{args.ds_name}/0.1.0",
            max_episodes_per_file=1,
            step_metadata_info=step_metadata_info,
        )

        env.set_step_metadata({"language_text": "pick up the can"})

    env = ConvertState2Proprio(env)
    env = ResizeObsImageWrapper(
        env, resize_size={"image_primary": (256, 256), "image_wrist": (128, 128)}
    )

    # # add wrappers for history and "receding horizon control", i.e. action chunking
    env = HistoryWrapper(env, horizon=2)
    # env = HistoryWrapper(env, horizon=1)
    env = TemporalEnsembleWrapper(env, 4)
    # env = RHCWrapper(env, exec_horizon=4)

    # NOTE: we are using bridge_dataset's statistics for default normalization
    # wrap env to handle action/proprio normalization -- match normalization type to the one used during finetuning
    # print(model.dataset_statistics[-1])\
    # print(model.dataset_statistics)
    # print(type(model.dataset_statistics['action']['max']))
    # stats = model.dataset_statistics[-1]
    # print(stats)
    # for k in stats:
    #     if type(stats[k]) == dict:
    #         for j in stats[k]:
    #                 stats[k][j] = np.array(stats[k][j])
    #     else:
    #         stats[k] = np.array(stats[k])

    env = UnnormalizeActionProprio(
            env, model.dataset_statistics, normalization_type="normal"
    )

    # running rollouts
    seed = 0
    sampling_rng = jax.random.key(seed)

    # store which trajecs are succesfull
    success_trajecs = {}

    grasp_detected = False
    grasp_wait_counter = 0

    max_len_of_trajec = 100

    for j in range(200):
        one_traj_data = np.array([])
        if args.no_reset_pose:
            obs, info = env.reset(reset_pose=bool(False))
        else:
            obs, info = env.reset(reset_pose=bool(True))

        # create task specification --> use model utility to create task dict with correct entries
        language_instruction = [args.text_cond]
        task = model.create_tasks(texts=language_instruction)
        # task = model.create_tasks(goals={"image_primary": img})   # for goal-conditioned

        episode_return = 0.0
        for i in range(max_len_of_trajec):
            if args.show_img:
                img = obs["image_primary"][0]
                cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img2 = obs["image_wrist"][0]
                cv2.imshow("image_wrist", cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
                cv2.waitKey(10)
                # capture "r" key and reset
                if cv2.waitKey(100) & 0xFF == ord("s") and i > 10:
                    if recording_data:
                        success_trajecs[j] = 1
                        print("success")
                        with open(
                            success_trajecs_file_name, "wb"
                        ) as f:
                            pickle.dump(success_trajecs, f)
                    break

                if cv2.waitKey(100) & 0xFF == ord("f") and i > 10:
                    if recording_data:
                        success_trajecs[j] = 0
                        print("failure")
                        with open(
                            success_trajecs_file_name, "wb"
                        ) as f:
                            pickle.dump(success_trajecs, f)
                    break

            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            rng_ac, sampling_rng = jax.random.split(sampling_rng)
            # no_random = jax.random.key(0)
            print("rng action seed", rng_ac)
            actions = model.sample_actions(
                jax.tree_map(lambda x: x[None], obs), task, rng=rng_ac
            )

            actions = actions[0]
            print("performing action: ", actions)
            print(f"Step {i} with action size of {len(actions)}")
            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            # one_traj_data = np.append(one_traj_data, {"action" : actions})
            # actions = actions.at[:, 3:6].set(0)

            obs, reward, done, trunc, info = env.step(actions)
            latest_ac = actions[-1] # last action in the chunk is the latest action

            # detect grasp and wait 10 time steps after to detect if valid grasp
            print("proprio", obs["proprio"])
            print(latest_ac)
            if latest_ac[6] <= 0.5 and not grasp_detected:
                if recording_data:
                    grasp_detected = True
                    
            success_trajecs_file_name = f"{args.ds_name}/{args.ds_name}/0.1.0/success_trajecs.pkl"

            # TODO (YL): cleanup this impl, it is too messy
            # if grasp detected
            if grasp_detected:
                if grasp_wait_counter < 5:
                    grasp_wait_counter += 1
                else:
                    if latest_ac[6] <= 0.5:
                        final_vote = task_success_classifier(obs)

                        if final_vote:
                            success_trajecs[j] = 1
                            print("success")
                            with open(
                                success_trajecs_file_name,
                                "wb",
                            ) as f:
                                pickle.dump(success_trajecs, f)
                        else:
                            success_trajecs[j] = 0
                            print("failure")
                            with open(
                                success_trajecs_file_name,
                                "wb",
                            ) as f:
                                pickle.dump(success_trajecs, f)

                        grasp_detected = False
                        grasp_wait_counter = 0
                        break
                    else:
                        success_trajecs[j] = 0
                        print("failure")
                        with open(
                            success_trajecs_file_name,
                            "wb",
                        ) as f:
                            pickle.dump(success_trajecs, f)
                        grasp_detected = False
                        grasp_wait_counter = 0
                        break

            if i == max_len_of_trajec - 1:
                if recording_data:
                    success_trajecs[j] = 0
                    print("failure")
                    with open(
                        success_trajecs_file_name, "wb"
                    ) as f:
                        pickle.dump(success_trajecs, f)
                    grasp_detected = False
                    grasp_wait_counter = 0
                break

            print(
                "grasp detected and counter: ", grasp_detected, " ", grasp_wait_counter
            )

            print("obs debug: ")
            print(actions.dtype)
            print("-" * 50)

            episode_return += reward

        # np.save('octo_eval_traj' + str(i), one_traj_data)
        print(f"Episode return: {episode_return}")
