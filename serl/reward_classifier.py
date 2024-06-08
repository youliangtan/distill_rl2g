import requests
import cv2
import numpy as np
from typing import Dict
from PIL import Image
import base64
import gym
import time
from gym import Env, spaces
import gym
import numpy as np


print_yellow = lambda x: print("\033[93m {}\033[00m".format(x))
print_green = lambda x: print("\033[92m {}\033[00m".format(x))
print_red = lambda x: print("\033[91m {}\033[00m".format(x))

sigmoid = lambda x: 1 / (1 + np.exp(-x))


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def chatgpt_task_success_classifier(
        obs: Dict[str, np.ndarray],
        prompt: str,
        num_votes: int = 3,
    ) -> bool:
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

    # w_img_to_check = obs["image_wrist"][0]
    # w_img_to_check = cv2.cvtColor(w_img_to_check, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("wrist_grasp_detection_image.jpg", w_img_to_check)

    p_image_path = "primary_grasp_detection_image.jpg"
    w_image_path = "wrist_grasp_detection_image.jpg"
    # Getting the base64 string
    base64_image_prim = encode_image(p_image_path)
    # base64_image_wrist = encode_image(w_image_path)

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
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_prim}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

                    #     {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:image/jpeg;base64,{base64_image_wrist}"
                    #     },
                    # },

    votes = []

    for i in range(num_votes):
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


class PickUpObjectBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminal, trunc, info = self.env.step(action)
        # check when the gripper is close
        print("state: ", obs["state"][-1])
        state = obs["state"][-1]

        # TODO: 0.7 is arbitrary, should be replaced with the actual value
        if 0.5 < state[-1] < 0.7 and \
            0.1 < state[2] < 0.2: # z-axis
            reward = 1
            terminal = True
            print("Task is successful reset")
        else:
            reward = 0
        return obs, reward, terminal, trunc, info


class GPTBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._steps_since_grasp = 0
        self._steps_since_grasp_to_initiate_chatgpt = 2

    def step(self, action):
        obs, reward, terminal, trunc, info = self.env.step(action)       

        # check when the gripper is close
        state = obs["state"][-1]
        print("state: ", state)
        if state[2] > 0.1:
            reward -= 0.01*state[2]

        if state[-1] > 0.5:
            if state[2] < 0.02:
                reward += 0.01
                self._steps_since_grasp += 1 # close
        else:
            self._steps_since_grasp = 0 # open

        if self._steps_since_grasp > self._steps_since_grasp_to_initiate_chatgpt:
            print_yellow("Initiating chatgpt to determine task success \n\n")
            prompt = "I'm providing the third person and wrist camera images of the robot arm. Is the robot arm gripper grabbing the object? Please respond with true or false, one word and all lower case."
            is_success = chatgpt_task_success_classifier(obs, prompt)
            if is_success:
                print_green("Task is successful")
                terminal = True
                reward = 1
            else:
                print_red("Task is not successful")
                reward = 0
            self._steps_since_grasp = 0
        return obs, reward, terminal, trunc, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FWBWFrontCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    # def __init__(self, env):
    #     super().__init__(env)
    #     self._steps_since_grasp = 30
    #     self._steps_since_grasp_to_initiate_chatgpt = 10

    # def step(self, action):
    #     obs, reward, terminal, trunc, info = self.env.step(action)       

    #     # check when the gripper is close
    #     print("state: ", obs["state"][-1])
        
    #     state = obs["state"][-1]
    #     if state[-1] > 0.5:
    #         self._steps_since_grasp = 0 # open
    #     else:
    #         reward -= 0.02
    #         if state[2] < 0.01:
    #             self._steps_since_grasp += 1 # close

    #     if self._steps_since_grasp > self._steps_since_grasp_to_initiate_chatgpt:
    #         print_yellow("Initiating chatgpt to determine task success \n\n")
    #         prompt = "I'm providing the third person and wrist camera images of the robot arm. Is the robot arm grabbing the coke can? Please respond with true or false, one word and all lower case."
    #         is_success = chatgpt_task_success_classifier(obs, prompt)
    #         if is_success:
    #             print_green("Task is successful")
    #             terminal = True
    #             reward = 1
    #         else:
    #             print_red("Task is not successful")
    #             reward = 0
    #         self._steps_since_grasp = 0
    #     return obs, reward, terminal, trunc, info

    # def reset(self, **kwargs):
    #     return self.env.reset(**kwargs)
    
    def __init__(self, env: Env, fw_reward_classifier_func, bw_reward_classifier_func):
        # check if env.task_id exists
        # assert hasattr(env, "task_id"), "fwbw env must have task_idx attribute"
        # assert hasattr(env, "task_graph"), "fwbw env must have a task_graph method"

        super().__init__(env)
        self.reward_classifier_funcs = [
            fw_reward_classifier_func,
            bw_reward_classifier_func,
        ]

    # def task_graph(self, obs):
    #     """
    #     predict the next task to transition into based on the current observation
    #     if the current task is not successful, stay in the current task
    #     else transition to the next task
    #     """
    #     success = self.compute_reward(obs)
    #     if success:
    #         return (self.task_id + 1) % 2
    #     return self.task_id

    def compute_reward(self, obs):
        reward = self.reward_classifier_funcs[0](obs).item() # TODO: replace with actual task ID
        return (sigmoid(reward) >= 0.5) * 1

    def step(self, action):
        # obs, rew, done, truncated, info = self.env.step(action)
        # rew = self.compute_reward(self.env.get_front_cam_obs())

        obs, reward, terminal, trunc, info = self.env.step(action)

        # check when the gripper is close
        print("state: ", obs["state"][-1])
        state = obs["state"][-1]

        # reward = self.compute_reward(obs['image_primary'])
        reward = self.compute_reward(obs)

        # done = done or rew
        return obs, reward, terminal, trunc, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
