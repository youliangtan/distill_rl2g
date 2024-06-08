# Distill_rl2g

Project to distill RL policies to generalist model

Forked from: https://github.com/RagsToHrishes/WiperBot

## Dependencies

**Learning code**
- https://github.com/rail-berkeley/serl
- https://github.com/rail-berkeley/manipulator_gym
- https://github.com/youliangtan/agentlace
- https://github.com/rail-berkeley/oxe_envlogger
- https://github.com/octo-models/octo


## Run the ViperX robot

This requires the installation of ros with viperx. Follow: https://docs.trossenrobotics.com/interbotix_xsarms_docs/ros_interface/ros1/software_setup.html

```bash
cd launch
roslaunch wiper_bot.launch robot_model:=vx300s
```

```bash
# external pkg
cd manipulator_gym
python manipulator_server.py --viperx --non_blocking --resize_img 128 128
```

## Run RL Policy with SERL

### Training the Classifier

Requires Oculus Reader. Install: https://github.com/rail-berkeley/oculus_reader

1. Collect trajectories of the task using teleoperation.

```bash
cd utils/

# publish the oculus data
python vr_data_collection.py --oculus_publisher True

# on the second terminal
python record_single_trajectory.py --robot_ip 100.96.12.13 --label positive
```

2. Train the classifier with the data

Download pretrained resnet:

```bash
wget https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl
```

```bash
python train_reward_classifier.py \
 --negative_demo_paths serl_transitions_negative_2024-05-30_23-32-36.pkl \
 --negative_demo_paths serl_transitions_negative_2024-05-30_23-34-25.pkl \
 --negative_demo_paths serl_transitions_negative_2024-05-30_23-36-06.pkl \
 --positive_demo_paths serl_transitions_positive_2024-05-30_23-38-42.pkl \
 --positive_demo_paths serl_transitions_positive_2024-05-30_23-41-11.pkl \
 --num_epochs 100
```

### Collect Human expert demonstations

This is for RLPD, 20 trajectories are collected for each task.

```bash
cd utils/

# publish the oculus data
python vr_data_collection.py --oculus_publisher True

# on the second terminal
python vr_data_collection.py --robot_ip 100.96.12.13 --show_img True --rlds_output demo_bin_relocate_20_2025-05-14
```

### Run SERL

This is to run serl code on the robot. First, the provide the manipulator_ip as the ip of the robot. Then, run the following commands on the robot and the learner node.

RLDS is enabled by providing the `--preload_rlds_path` argument to the learner node. The path should be the path to the RLDS dataset.

learner node
```bash
python viperx_drq.py --batch_size 128 --checkpoint_period 5000 --checkpoint_path /hdd/serl_chkpts2/ --learner --preload_rlds_path /hdd/serl/task1_2jun_combine_fixbblock/ --reward_classifier_ckpt_path checkpoint_20
```

add ` --checkpoint_path /hdd/serl_chkpts/` to save/load checkpoints

Actor node
```bash
python viperx_drq.py --manipulator_ip 100.96.12.13 --actor --show_img --reward_classifier_ckpt_path checkpoint_20
```
