
import numpy as np
from absl import app, flags, logging

import sys
import cv2
import time

from pyquaternion import Quaternion
import argparse

from manipulator_gym.manipulator_env import ManipulatorEnv
from manipulator_gym.interfaces.base_interface import ManipulatorInterface
from manipulator_gym.interfaces.interface_service import ActionClientInterface
import manipulator_gym.utils.transformation_utils as tr
from manipulator_gym.utils.gym_wrappers import ResizeObsImageWrapper, ClipActionBoxBoundary
from oculus_reader.reader import OculusReader

# use agentlace for fancy broadcasting of controller
from agentlace.action import ActionClient, ActionServer, ActionConfig

# factor of oculus linear movement to robot linear movement
LINEAR_ACTION_FACTOR = 1.1
# how much to clip the linear action
CLIP_LINEAR_ACTION = 0.04
# how much to rotate every step
ANGULAR_ACTION_PER_STEP = 0.06

TIME_STEP = 0.1

print_red = lambda x: print("\033[91m {}\033[00m".format(x))

class TeleopDataCollector:
    def __init__(
            self,
            oculus_reader,
            env,
            resize_img,
            shared_controller,
            log_dir=None,
            lang_prompt="doing something",
        ) -> None:
        """
        To control the robot using the oculus and collect data
        """
        self.ref_point_set = False
        self.gripper_open = False
        self.reference_vr_transform = None
        self.reference_robot_transform = None

        self.a_button_press = False
        self.a_button_down = False
        self.b_button_press = False
        self.b_button_down = False
        self.RTr_button_press = False
        self.RTr_button_down = False
        self.RJ_button_press = False
        self.RJ_button_down = False
        self.intialize = False

        self.oculus_reader = oculus_reader
        self.env = env

        if resize_img:
            self.env = ResizeObsImageWrapper(
                self.env,
                resize_size={"image_primary": (256, 256), "image_wrist": (128, 128)}
            )

        self.shared_controller = shared_controller

        self.one_traj_data = np.array([]) # what is this for?
        self.traj_num = 0
        self.log_dir = log_dir  # this will also track if the data collection is enabled or not
        self.current_language_text = lang_prompt

        # if log_dir is not None, then we will log the data as RLDS
        # Uses: https://github.com/rail-berkeley/oxe_envlogger
        if log_dir is not None:
            import tensorflow_datasets as tfds
            from oxe_envlogger.envlogger import OXEEnvLogger

            step_metadata_info = {'language_text': tfds.features.Text(
                doc="Language embedding for the episode.")
            }
            # step_metadata_info = {} # TODO: possibly fix
            self.env = OXEEnvLogger(
                env,
                dataset_name="serl_demos",
                directory=log_dir,
                max_episodes_per_file=1,
                step_metadata_info=step_metadata_info,
            )

    def oculus_to_robot(self, current_vr_transform):
        z_rot = tr.RpToTrans(Quaternion(
            axis=[0, 0, 1], angle=-np.pi / 2).rotation_matrix)
        x_rot = tr.RpToTrans(Quaternion(
            axis=[1, 0, 0], angle=np.pi / 2).rotation_matrix)
        current_vr_transform = z_rot.dot(x_rot).dot(current_vr_transform)
        return current_vr_transform
    
    def get_action(self) -> tuple:
        """
        get actions from the vr controller
        
        return action, ep_done, buttons
        """
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
        ep_done = False
        # print("buttons: ", buttons)
        # print("transformations: ", transformations)
        if "r" in transformations:
            vr_transform = transformations['r']

        if not self.intialize:
            self.ref_point_set = True
            self.gripper_open = True # Default open?

            self.reference_vr_transform = self.oculus_to_robot(vr_transform)
            self.initial_vr_offset = tr.RpToTrans(
                np.eye(3), self.reference_vr_transform[:3, 3])
            self.reference_vr_transform = tr.TransInv(
                self.initial_vr_offset).dot(self.reference_vr_transform)
            self.intialize = True

        if len(buttons.keys()) <= 0:
            return

        if buttons["A"] and not self.a_button_down:
            self.a_button_down = True

        if self.a_button_down and not buttons["A"]:
            self.a_button_press = True
            self.a_button_down = False

        if buttons["B"] and not self.b_button_down:
            self.b_button_down = True

        if self.b_button_down and not buttons["B"]:
            self.b_button_press = True
            self.b_button_down = False

        if buttons["RJ"]:
            ep_done = True

        if (self.b_button_press):
            if len(self.one_traj_data) > 0:
                # np.save('traj_' + str(self.traj_num), self.one_traj_data)
                self.traj_num += 1
                self.one_traj_data = np.array([])

            # input("Paused, press enter to continue")
            print("Starting a Trajectory")
            ep_done = True
            self.ref_point_set = True
            self.gripper_open = True # Default open?

            self.reference_vr_transform = self.oculus_to_robot(vr_transform)
            self.initial_vr_offset = tr.RpToTrans(
                np.eye(3), self.reference_vr_transform[:3, 3])
            self.reference_vr_transform = tr.TransInv(
                self.initial_vr_offset).dot(self.reference_vr_transform)

            self.b_button_press = False

        # Handling reference positions and starting and stopping VR using the A button
        if self.a_button_press and not self.ref_point_set:
            print("Reference point set, motion VR enabled")
            self.ref_point_set = True

            self.reference_vr_transform = self.oculus_to_robot(vr_transform)
            self.initial_vr_offset = tr.RpToTrans(
                np.eye(3), self.reference_vr_transform[:3, 3])
            self.reference_vr_transform = tr.TransInv(
                self.initial_vr_offset).dot(self.reference_vr_transform)

            self.a_button_press = False

        if self.a_button_press and self.ref_point_set:
            print("Reference point deactivated, motion VR disabled")
            self.ref_point_set = False
            self.reference_vr_transform = None
            self.initial_vr_offset = None
            self.a_button_press = False

        # Performing a step in the env if ref_point and VR enabled motion has been activated
        # Copy the most recent T_yb transform into a temporary variable
        # Calulating relative changes and putting into action
        if self.gripper_open:
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1])
        else:
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])

        """Will start collecting data if the reference point is set"""
        if self.ref_point_set:
            current_vr_transform = self.oculus_to_robot(vr_transform)
            delta_vr_transform = current_vr_transform.dot(
                tr.TransInv(self.reference_vr_transform))
            M_delta, v_delta = tr.TransToRp(delta_vr_transform)

            action[:3] = v_delta*LINEAR_ACTION_FACTOR
            self.reference_vr_transform = self.oculus_to_robot(vr_transform)

            # arm_eul =  ang.rotationMatrixToEulerAngles(ref_orientation_arm)
            # arm_eul[0] -= vr_eul[1]
            # arm_eul[1] -= vr_eul[0]
            # arm_eul[2] -= vr_eul[2]
            # T_yb[:3,:3] = ang.eulerAnglesToRotationMatrix(arm_eul)
            # self.update_T_yb()

        # Use button for orientation
        action[3] = np.clip(buttons['rightJS'][0], -ANGULAR_ACTION_PER_STEP, ANGULAR_ACTION_PER_STEP)
        action[4] = np.clip(buttons['rightJS'][1], -ANGULAR_ACTION_PER_STEP, ANGULAR_ACTION_PER_STEP)
        action[6] = 0 if buttons["RTr"] else 1  # gripper

        # Applying relative changes by stepping the environment
        # Define the range you want to bound the values to
        # Use np.clip() to bound the values within the specified range
        action[:3] = np.clip(action[:3], -CLIP_LINEAR_ACTION, CLIP_LINEAR_ACTION)

        assert len(action) == 7        

        if self.ref_point_set:
            action = action.astype(np.float32)
            return action, ep_done, buttons
        else:
            # action = np.zeros((7,), dtype=np.float32)
            # action[-1] = 1 if self.gripper_open else 0
            action = None
            print("No action taken")
            return action, ep_done, buttons

class OculusReaderListener:
    def __init__(
        self,
        host,
        port=5561,
        apply_yaw_correction: float = 0.0,
    ) -> None:
        _action_config = ActionConfig(
            port_number=port, broadcast_port=port + 1,
            action_keys=[], observation_keys=["transformations", "buttons"],
        )
        self._client = ActionClient(host, _action_config)
        self._client.register_obs_callback(self._msg_callback)

        # TODO: this might not be correct, need to debug this
        self.yaw_correction_matrix = tr.RpToTrans(
            Quaternion(axis=[1, 0, 0], angle=apply_yaw_correction).rotation_matrix
        )
        self._default_val = {
            "r": np.eye(4)
        }, {
            "A": False,
            "B": False,
            "RTr": False, # gripper, the pointer button
            "RJ": False, # end of everything, the joystick button
            "rightJS": [0, 0]
        }

    def _msg_callback(self, msg: dict):
        # print("new msg: ", msg)
        self._default_val = (
            msg["transformations"], msg["buttons"]
        )

    def get_transformations_and_buttons(self) -> tuple:
        transformations, buttons = self._default_val
        # print(self.yaw_correction_matrix)
        transformations["r"] = np.matmul(
            self.yaw_correction_matrix, transformations["r"]
        )  
        return transformations, buttons


class OculusReaderPublisher:
    def __init__(self, port=5561, interval=0.1) -> None:
        _action_config = ActionConfig(
            port_number=port, broadcast_port=port + 1,
            action_keys=[], observation_keys=["transformations", "buttons"],
        )
        self._server = ActionServer(_action_config, obs_callback=None, act_callback=None)
        self.reader = OculusReader()
        self._interval = interval
        self._server.start(threaded=True)

    def run(self):
        """This will run the server and broadcast the transformations and buttons"""
        while True:
            transformations, buttons = self.reader.get_transformations_and_buttons()

            # print red if transformations and buttons are dict
            if transformations is None or buttons is None:
                print_red("No data from the oculus, \
                    you might want to make sure the oculus is active")
                continue

            print("transformations: ", transformations)
            self._server.broadcast.broadcast({
                "transformations": transformations,
                "buttons": buttons
            })
            time.sleep(self._interval)


class DummyOculusReader:
    """This is a dummy oculus reader to mimic the oculus reader for testing purposes"""

    def __init__(self, num_steps=10, num_of_episodes=3):
        self.num_steps = num_steps
        self.num_of_episodes = num_of_episodes
        self.current_step = self.num_steps  # to trigger the reset
        self.current_episode = 0
        self._end_of_logging = False  # some bad hack to end the logging

    def get_transformations_and_buttons(self):
        """
        This will mimic the oculus reader and return the transformations and buttons
        call the reset every start and end the episode after max steps is reached,
        then call end the reader when the num of episodes is reached
        """
        default_val = {
            "r": np.eye(4)
        }, {
            "A": False,
            "B": False,
            "RTr": False,
            "RJ": False,
            "rightJS": [0, 0]
        }

        if self.current_step == self.num_steps:
            print("mimic starting new trajectory")
            self.current_step = 0
            default_val[1]["B"] = True
            self.current_episode += 1
            return default_val

        if self.current_episode == self.num_of_episodes:
            print("mimic end of logging")
            if not self._end_of_logging:
                default_val[1]["RJ"] = True
            self._end_of_logging = True
            return default_val

        self.current_step += 1
        return default_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--robot_ip", help="IP address of the robot server.",
                    default="localhost", type=str)
    parser.add_argument("--oculus_server_ip", help="oculus server ip",
                    default="localhost", type=str)
    parser.add_argument("--show_img", help="Whether to visualize the images or not.",
                    default= False, type=bool)
    parser.add_argument("--test", help="Whether to test the data collection or not.",
                    default= False, type=bool)
    parser.add_argument("--resize_img", help="Whether to resize the images or not.",
                    default= False, type=bool)
    parser.add_argument("--rlds_output", help="RLDS output",default= None,type=str)
    parser.add_argument("--ds_name", help=" ds_name", default="task1", type=str)
    parser.add_argument("--lang_prompt", help="lang_prompt", default="doing something", type=str)
    parser.add_argument("--use_keyboard", help="use_keyboard", action="store_true")
    parser.add_argument("--oculus_publisher", help="Whether to publish oculus",action="store_true")
    parser.add_argument("--task", help="task", default= "task1", type=str)
    args = parser.parse_args()

    if args.test:
        env = ManipulatorEnv(manipulator_interface=ManipulatorInterface())
        reader = DummyOculusReader(num_of_episodes=40)
    elif args.oculus_publisher:
        # print("Starting Oculus Reader Publisher on port: ", args.port)
        reader = OculusReaderPublisher()
        reader.run()
        exit()
    elif args.use_keyboard:
        print("Using Keyboard")
    else:
        from oculus_reader.reader import OculusReader
        from task_configs import get_task_config

        config = get_task_config(args.task)

        env = ManipulatorEnv(
            manipulator_interface=ActionClientInterface(host=args.robot_ip),
            use_wrist_cam=True,
        )
        env = ClipActionBoxBoundary(env, workspace_boundary=config.workspace_boundary)

        if args.oculus_server_ip:
            print("Connecting to Oculus Server, IP: ", args.oculus_server_ip)
            reader = OculusReaderListener(host=args.oculus_server_ip)
        else:
            print("Using Native Oculus Reader")
            reader = OculusReader()

    # this will dictate the size of the image that will be logged
    if args.resize_img:
        env = ResizeObsImageWrapper(
            env,
            resize_size={"image_primary": (256, 256), "image_wrist": (128, 128)}
        )

    data_collector = TeleopDataCollector(
        oculus_reader=reader,
        env=env,
        shared_controller=False, # TODO: possibly change?
        resize_img=args.resize_img,
        log_dir=f"{args.rlds_output}/{args.ds_name}/0.1.0",
        lang_prompt=args.lang_prompt,
    )

    stopDataCollection = False
    while not stopDataCollection:
        if args.show_img:
            obs = data_collector.env.obs()
            img = obs["image_primary"]
            cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.waitKey(10)
            # img = obs["image_wrist"]
            # cv2.imshow("image_wrist", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.waitKey(10)

        action, ep_done, buttons = data_collector.get_action()
        all_done = buttons["RJ"]

        if action is None:
            print("No action taken, paused...")
            continue

        if all_done:
            print("Stopping data collection")
            data_collector.env.close()
            break

        if ep_done:
            data_collector.env.reset(reset_pose=True, target_state=config.reset_pose)
            tmp = input("Paused, press enter to continue")

        assert len(action) == 7
        if data_collector.log_dir is not None:
            data_collector.env.set_step_metadata(
                {"language_text": data_collector.current_language_text})

        # cast action to float32 TODO: better and more robust way in oxeenvlogger
        action = action.astype(np.float32)
        data_collector.env.step(action)
        time.sleep(TIME_STEP)
