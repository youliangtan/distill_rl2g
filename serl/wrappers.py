import gym
from gym import Env
import numpy as np
from typing import Optional


def sigmoid(x): return 1 / (1 + np.exp(-x))


def print_blue(x): return print(f"\033[94m{x}\033[0m")


##############################################################################

class MaxEpsLengthWrapper(gym.Wrapper):
    def __init__(self, env, eps_length: int):
        super().__init__(env)
        self.eps_length = eps_length
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.current_step += 1
        observation, reward, terminate, truncate, info = self.env.step(action)
        if self.current_step > self.eps_length:
            print("reached max eps length, truncating episode")
            truncate = True
        return observation, reward, terminate, truncate, info


##############################################################################

class BinaryRewardClassifierWrapper(gym.Wrapper):
    """
    Compute reward with custom binary reward classifier fn
    """

    def __init__(
        self,
        env: Env,
        reward_classifier_func,
        terminate_on_n_reward: int = 1, # not in used (NOTE: maybe can introduced to prevent false positive)
    ):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            logit = self.reward_classifier_func(obs).item()
            return (sigmoid(logit) >= 0.5) * 1, False
        return 0, False

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        bin_rew, r_done = self.compute_reward(obs)
        done = done or r_done
        rew += bin_rew
        return obs, rew, done, truncated, info

##############################################################################

class GoToRewardWrapper(gym.Wrapper):
    def __init__(self,
                 env: Env,
                 target_pos: np.ndarray,
                 distance_threshold: float = 0.03,
                 with_gripper: bool = True,
                 gripper_open: bool = True,
                 reward_factor: float = 1.0,
                 dense_reward: bool = True,
                 ):
        super().__init__(env)
        self.target_pos = target_pos
        self.distance_threshold = distance_threshold
        self.reward_factor = reward_factor
        self.gripper_open = gripper_open
        self.dense_reward = dense_reward
        self.with_gripper = with_gripper
        print(f"reward wrapper: target_pos: {target_pos}")

    def compute_reward(self, obs, action):
        state = obs["state"][-1]
        dist = np.linalg.norm(state[:3] - self.target_pos)
        rew = -1.0
        done = False

        if np.abs(dist) < self.distance_threshold:
            print("reached target position, call done")
            
            # check gripper to see if we are done
            if self.with_gripper:
                if self.gripper_open == state[-1] > 0.5:
                    rew += 1.0
                    done = True
                else:
                    # gripper havent reached the desired state
                    pass
            else:
                rew += 1.0  # assume net is 0.0
                done = True

        if self.dense_reward:
            rew -= dist*self.reward_factor
        return rew, done

    def step(self, action):
        """Duplicate the reward logic from the reward wrapper"""
        obs, rew, done, truncated, info = self.env.step(action)
        r_rew, r_done= self.compute_reward(obs, action)
        rew += r_rew
        done = done or r_done
        return obs, rew, done, truncated, info

##############################################################################


class FancyRewardClassifierWrapperWithGripper(gym.Wrapper):
    """
    Compute reward with custom binary reward classifier fn
    with custom logic
    """

    def __init__(
        self,
        env: Env,
        reward_classifier_func,
        terminate_on_n_reward: int = 1,
        target_z: Optional[float] = None,
        target_z_lift: Optional[float] = None,
        gripper_penalty: Optional[float] = 0.1,
    ):
        """
        NOTE: use sparse reward when target_z_lift is provided
        """
        super().__init__(env)
        if reward_classifier_func is None:
            raise ValueError("reward_classifier_func cannot be None")

        self.reward_classifier_func = reward_classifier_func
        self.terminate_on_n_reward = terminate_on_n_reward
        self._target_z = target_z
        self._target_z_lift = target_z_lift # v2
        self._gripper_penalty = gripper_penalty # v3
        
        self._prev_cmd_gripper_open = True

    def compute_classifier_reward(self, obs):
        if self.reward_classifier_func is not None:
            logit = self.reward_classifier_func(obs).item()
            prob = sigmoid(logit)
            # print_blue(f"prob: {prob}")
            # return (prob >= 0.5) * 1
            return prob
        return 0.0

    def compute_reward(self, obs, action):
        if self._gripper_penalty:
            return self.compute_reward_v3(obs, action)
        elif self._target_z_lift:
            return self.compute_reward_v2(obs, action)
        else:
            return self.compute_reward_v1(obs, action)

    def compute_reward_v1(self, obs):
        rew = 0.0
        prob = self.compute_classifier_reward(obs)
        # rew += prob
        bin_rew = (prob >= 0.5) * 1

        state = obs["state"][-1]
        is_gripper_close = True if state[-1] < 0.55 else False

        # ensure gripper is open when bin_rew is 0 ( not grabbing )
        if bin_rew == 0 and is_gripper_close:
            rew -= 0.02

        # we will terminate the episode if we get n consecutive rewards
        # with true in reward classifier and gripper is close
        if bin_rew == 1 and is_gripper_close:
            # print("true from reward classifier and gripper is close")
            bin_rew = 2.0

        if self._target_z is not None and not is_gripper_close:
            # use z distance as reward when gripper is open
            dist = np.abs(state[2] - self._target_z)
            rew -= dist*0.05
        #     print(f"dist: {dist}")
        # print(f"bin_rew: {bin_rew}, gripper: {is_gripper_close}, gripper state: {state}")

        rew += bin_rew
        done = False
        return rew, done

    def compute_reward_v2(self, obs, action, debug=False):
        """
        Provide a reward when 
        the reward classifier is true and
        the gripper is close and 
        the z axis is greater than the target z lift
        """
        prob = self.compute_classifier_reward(obs)
        bin_rew = (prob >= 0.5) * 1

        state = obs["state"][-1]
        # NOTE: use 0.6 for gripper close since gripper might be grabbing something
        is_gripper_close = True if state[-1] < 0.55 else False
        z_axis = state[2]

        if debug:
            print(f"bin_rew: {bin_rew}, gripper: {is_gripper_close}, z_axis: {z_axis}")

        if bin_rew == 1 and is_gripper_close and z_axis >= self._target_z_lift:
            # print_blue("reached target z lift")
            rew , done = 1.0, True
        else:
            rew, done = 0.0, False
        return rew, done

    def compute_reward_v3(self, obs, action, debug=True):
        """
        An extension of sparse reward with gripper penalty
        """
        prob = self.compute_classifier_reward(obs)
        print_blue(f"prob: {prob}!!")
        bin_rew = (prob >= 0.5) * 1

        state = obs["state"][-1]
        # NOTE: use 0.6 for gripper close since gripper might be grabbing something
        is_gripper_close = True if state[-1] < 0.55 else False
        z_axis = state[2]

        rew = 0.0

        # penalize when there is a change in gripper command
        if action[-1] > 0.0 and not self._prev_cmd_gripper_open:
            rew -= self._gripper_penalty
            # print("penalize gripper open")
        elif action[-1] < 0.0 and self._prev_cmd_gripper_open:
            rew -= self._gripper_penalty
            # print("penalize gripper close")

        # update previous gripper command
        self._prev_cmd_gripper_open = action[-1] > 0.0

        if debug:
            print(f"bin_rew: {bin_rew}, gripper: {is_gripper_close}, z_axis: {z_axis}")

        # if we would like to lift the object
        if self._target_z_lift:
            if bin_rew == 1 and is_gripper_close and z_axis >= self._target_z_lift:
                print_blue("reached target z lift")
                rew += 1.0
                done = True
            else:
                done = False
        else:
            if bin_rew == 1 and is_gripper_close and z_axis <= self._target_z:
                print_blue("reached target z")
                done = True
                rew += 1.0
            else:
                done = False
        return rew, done

    def step(self, action):
        """Note action is normalized between -1 and 1"""
        obs, rew, done, truncated, info = self.env.step(action)
        r_rew, r_done= self.compute_reward(obs, action)
        rew += r_rew
        done = done or r_done
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        self._prev_cmd_gripper_open = True # gripper is open during reset
        return self.env.reset(**kwargs)

##############################################################################


class ResizeEnvAction(gym.Wrapper):
    """Convert 7 action space to N action space"""

    def __init__(self,
                 env: Env,
                 action_dim: int,
                 with_gripper: bool = True,
                 eef_displacement: float = 0.02,
                 ):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-eef_displacement, high=eef_displacement,
            shape=(action_dim,), dtype=np.float32
        )

        # gripper has a different range
        if with_gripper:
            self.action_space.low[-1] = 0.0
            self.action_space.high[-1] = 1.0

        self.action_dim = action_dim
        self.with_gripper = with_gripper

    def step(self, action):
        assert len(action) == self.action_dim
        print("action: ", action)
        mod_action = np.zeros(7, dtype=np.float32)

        if self.with_gripper:
            mod_action[:self.action_dim - 1] = action[:self.action_dim - 1]
            mod_action[-1] = action[-1]
        else:
            mod_action[:self.action_dim] = action[:self.action_dim]
        return self.env.step(mod_action)
