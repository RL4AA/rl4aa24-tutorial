# MAML for AWAKE

## Overview

- Implementation of Meta RL on the AWAKE electron line experiment at CERN
- Starting from a simulation covering a variation of MDPs
- The agent is prepared to learn as fast as possible on a realization of a specific MDP
- The code was benchmarked on the real machine

The code is based on:

```latex
@misc{deleu2018mamlrl,
  author = {Tristan Deleu},
  title = {{Model-Agnostic Meta-Learning for Reinforcement Learning in PyTorch}},
  note = {Available at: <https://github.com/tristandeleu/pytorch-maml-rl}>,
  year = {2018}
}
```

- How to use:
  - For meta-training run `train.py` and settings are adjustable in `awake/config.json` and `configs/awake.yaml`
  - For test training run `test.py`
  - For testing of learned policies run `run_maml.py`

## Getting started

The code is tested on Python 3.9 and 3.10.

You can either use _conda_ or _venv_ to install the required packages.

### Using Conda

If you have other python versions installed, we suggest creating a new virtual environment with conda

```bash
conda env create -f environment.yml
```

This should create an environment named `rl-tutorial` and install the necessary packages inside.

Afterwards, activate the environment using

```bash
conda activate rl-tutorial
```

### Using venv

_If you don't have conda installed:_

Alternatively, you can create the virtual env with

```bash
python venv -n rl-tutlrial
```

and activate the env with `$ source <venv>/bin/activate` (bash) or `C:> <venv>/Scripts/activate.bat` (Windows)

Then, install the packages with `pip` within the activated environment

```bash
python -m pip3 install -r requirements.txt
```

Afterwards, you should be able to run the provided scripts.

## Using the code

### Training PPO on a single task

First, we consider a single instance of the AWAKE tuning problem.
It then becomes a classical RL tuning task and can be solved using PPO.

With the virtual environment activated, run

```bash
python ppo.py --train
```

This will train a policy to solve the AWAKE problem using the PPO algorithm from stable-baselines3.

After training, a verification plot will be shown.

#### Evaluating the PPO agent on a different task

The PPO agent was trained only on _task_0_.
We can load the trained policy and evaluate it on other tasks, for example by running the following command

```bash
python ppo.py --test --task-id 2
```

### Adaptation from the random initial policy

The MAML algorithm is a two-step process, with an _outer-loop_ updating the _meta-policy_, and an _inner-loop_ (also called adaptation steps) to adapt the meta-policy to the individual tasks.

If we run only the inner-loop on a single task.

This is done by calling

```bash
python test.py
```

In the progress plots, you will see the agent is slowly improving.

### Looking at the meta-RL task

Now, let's run the meta-training.

```bash
python train.py
```

During the training, you can use the `run_update_training.py` to show some live updates of the training process.

After the meta-training, the meta-policy will be placed in a setting where it's easier to adapt to individual tasks.

We can verify this by running the `test.py` again, this time with the pre-trained meta-policy

```bash
python test.py --use-meta-policy
```

Now you should see that the agents behaves quite well initially, and still gets better after several adaption steps.

_Note_: To load the pre-trained meta-policy we provided, you can run it with

```bash
python test.py --use-meta-policy --num-batches 500 --policy awake/pretrained_policy.th --experiment-name test_me --experiment-type pretrained --task-ids 0 1 2 3 4 --plot-interval 100
```

It will load the policy from `awake/pretrained_policy.th` and adapt on the 5 verification tasks `[0,1,2,3,4]` for 500 batches, and save the results and progress to `awake/test_me/pretrained`.

Then you can run the following command to view the progress of the adaptation.

```bash
python read_out_train.py --experiment-name test_me --experiment-type pretrained
```

## Repository Structure

### Important files

The essential part to go through the tutorial, and to change the training behavior.

- `ppo.py`: trains an agent on one specific task of the AWAKE problem using PPO
- `train.py`: performs the meta-training on AWAKE problem
- `test.py`: performs the evaluation of the trained policy
- `configs/`: stores the yaml files for training configurations

### AWAKE Environment

This part is important if you want to understand what is happening in the AWAKE environment from a physics point of view.

- `maml_rl/envs/awake_steering_simulated.py` implements the AWAKE steering problem as a gymnasium environment.
- `maml_rl/envs/helpers.py` implement some helper classes for task sampling for the AWAKE environment.

### Utility files

This part is important if you want to change the behavior of progress logging, verification plots, etc.

- `policy_test.py` contains evaluation and verification functions.

### MAML Logic

This part is important if you want to have a deeper understanding of the MAML algorithm.

- `maml_rl/metalearners/maml_trpo.py` implements the _TRPO_ algorithm for the outer-loop.
- `maml_rl/policies/normal_mlp.py` implements a simple MLP policy for the RL agent.
- `maml_rl/utils/reinforcement_learning.py` implements the _Reinforce_ algorithm for the inner-loop.
- `maml_rl/samplers/` handles the sampling of the meta-trajectories of the environment using the multiprocessing package.
- `maml_rl/baseline.py` A linear baseline for the advantage calculation in RL.
- `maml_rl/episodes.py` A custom class to store the results and statistics of the episodes for meta-training.

## Contacts and Contributing

Don't hesitate to ask if you have any further questions: simon.hirlaender(at)plus(dot)ac(dot)at
