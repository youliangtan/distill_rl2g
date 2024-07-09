# Distill_rl2g

Project to distill RL policies to generalist model

Forked from: https://github.com/RagsToHrishes/WiperBot

## Dependencies

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

---

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
python vr_data_collection.py --oculus_publisher

# on the second terminal
python vr_data_collection.py --robot_ip 100.96.12.13 --show_img True --rlds_output DATASET_DIR_NAME --lang_prompt "do something special"
```

### Run SERL

This is to run serl code on the robot. First, the provide the `manipulator_ip` as the ip of the robot. Then, run the following commands on the robot and the learner node.

RLDS is enabled by providing the `--preload_rlds_path` argument to the learner node. The path should be the path to the RLDS dataset.

**Eval BC Policy**

First we train the BC policy on the expert demonstrations.

```bash
python bc_policy.py --batch_size 128 --preload_rlds_path /hdd/serl/serl_task1_combine_18jun/ --checkpoint_path /hdd/serl_bc_chkpt_3jul/
```

Then we evaluate the policy on the robot.

```bash
python bc_policy.py --manipulator_ip 100.96.12.13 --show_img \
--checkpoint_path /hdd/serl_bc_chkpt_3jul/ \
--eval_checkpoint_step 20000
```

**learner node**

We can run the learner without the robot.

```bash
python viperx_drq.py --batch_size 256  --learner \
--checkpoint_period 5000 --checkpoint_path /hdd/serl_chkpts2/ \
--reward_classifier_ckpt_path checkpoint_20 \
--preload_rlds_path /hdd/serl/serl_task1_combine_18jun/ \
--preload_online_rlds_path /hdd/serl/task1_online_data_17jun/ \
# --log_rlds_path /hdd/serl/task1_online_data_17jun_dense/ \
```

add ` --checkpoint_path /hdd/serl_bc_chkpt_3jul/` to save/load checkpoints

**Actor node**
```bash
python viperx_drq.py --actor \
--manipulator_ip 100.96.12.13 --show_img --reward_classifier_ckpt_path checkpoint_20

# provide additional args for ibrl
--bc_chkpt_path /hdd/serl_bc_chkpt/ --bc_chkpt_step 20000
```

To evaluate the model on the actor, add ` --checkpoint_path /hdd/serl_chkpts/` to load checkpoints. 

Provide `--log_rlds_path DIR_NAME` to save the online trajectories out as RLDS data.

Provide online rlds as `--preload_online_rlds_path DIR_NAME` to load the online rlds data.

---

## Run Octo Finetuning

We will now finetune with RL Rollouts generated from the trained DRQ model.

Now finetune the model using the generated RLDS dataset
```bash
cd octo
python scripts/finetune.py --config=../bc/viperx_finetune_config.py --config.pretrained_path=hf://rail-berkeley/octo-small
```

Then evaluate the model

```bash
python octo_eval.py --checkpoint_path MODEL_PATH \
--ip IP_ADDRESS --show_img --text_cond "put the banana on the plate"
```

---

# Miscs

To reshard the dataset, use the following command:

```bash
python oxe_envlogger/reshard_rlds.py --h # provide args
```


To read the rlds data for debugging
```bash
python manipulator_gym/read_rlds.py --h # provide args
```

Further robot related details, checkout: https://docs.google.com/document/d/1ka_eFiXbLxi1iIjKdj6b3So-xJhg2Cjd0j9N4w20w8Y/edit?pli=1
