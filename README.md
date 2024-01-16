# Neighborhood Mixup Experience Replay (NMER)
---------------------------------------------------------------------------------------------------------------------------------------------------------
[Ryan Sander](https://scholar.google.com/citations?user=7B6apiIAAAAJ&hl=en)<sup>1</sup>, [Wilko Schwarting](https://scholar.google.com/citations?hl=en&user=YI1EqBoAAAAJ)<sup>1</sup>, [Tim Seyde](https://scholar.google.com/citations?hl=en&user=FJ7ILzkAAAAJ)<sup>1</sup>, [Igor Gilitschenski](https://scholar.google.com/citations?hl=en&user=Nuw1Y4oAAAAJ)<sup>2,3</sup>, [Sertac Karaman](https://scholar.google.com/citations?hl=en&user=Vu-Zb7EAAAAJ)<sup>4</sup>, [Daniela Rus](https://scholar.google.com/citations?hl=en&user=910z20QAAAAJ)<sup>1</sup>

1 - MIT CSAIL, 2 - University of Toronto, 3 - Toyota Research Institute, 4 - MIT LIDS

**[Paper (L4DC 2022)](https://proceedings.mlr.press/v168/sander22a/sander22a.pdf)** | **[AirXv](https://arxiv.org/abs/2205.09117)** | **[Website](https://sites.google.com/view/nmer-drl)**

---------------------------------------------------------------------------------------------------------------------------------------------------------

![nmer_diagram](img/diagram.png)

## Overview - What is NMER?
Sample efficiency is a crucial component of successful deep reinforcement learning, particularly for high-dimensional robotics applications. NMER is a novel replay buffer technique designed for improving continuous control tasks that recombines previous experiences of deep reinforcement learning agents linearly through a simple geometric heuristic.

Namely, NMER improves sample efficiency of off-policy, model-free deep reinforcement learning algorithms by recombining transitions from the set of convex linear combinations of existing pairs of proximal samples.

## Installation
Installation can be done with either via `conda` or `pip`:

* Installing with `conda`: 

```conda env create -f env/environment.yml```

* Installing with `pip`:

```pip install -r env/requirements.txt```

To ensure you have installed all modules properly, please run `smoketest.py` in the virtual environment for this repository (e.g. with `conda` as follows):
```
conda activate interreplay
python3 smoketest.py
```

## MuJoCo Configuration
If you use OpenAI Gym MuJoCo environments, you will need to install MuJoCo and set the following path variable in your `~/.bashrc`. Note that MuJoCo can only be installed on Linux and MacOS operating systems. For more instructions on how to install MuJoCo, please follow the instructions [here](https://github.com/openai/mujoco-py).

Please make sure you set the following environment variable:

```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<user>/.mujoco/mujoco200/bin```

To obtain a free, one-month MuJoCo personal license, you can navigate to the page [here](https://www.roboti.us/license.html). **NOTE**: If you plan on running this framework in a HPC center with different nodes, you will need an institutional license to run jobs on multiple different nodes.


## Baseline Training with RLlib
Baseline training can be run with `ray[rllib]` through both the command-line interface and the Python API.

### Command-Line Interface
To run a baseline reinforcement learning algorithm on a given environment, you can do so via the command-line interface of `rllib`:
```
rllib train --run <algorithm> --env <environment>
```

For example, to run with Soft-Actor Critic (`SAC`):

```
# Runs CartPole-v0 environment with Soft Actor-Critic (SAC) Agent
rllib train --run SAC --env CartPole-v0
```

### Interpolated Replay Buffer Suite
For comparing the performance of NMER to other replay buffer approaches, we compare this algorithm to the following implemented [baseline custom replay buffers](./replay_buffers/baselines/):

1. Vanilla Replay Buffer
2. Prioritized Experience Replay (PER)
3. Continuous Transition (CT)
4. Surprisingly Simple Self-Supervision (S4RL)
5. Naive Mixup
6. Gaussian Noise


### Command-Line Interface
To run Bayesian Interpolated Experience Replay (BIER), you can do so by running 
`custom_training.py` with command-line arguments:

```
python3 custom_training.py <ARGS>
```
For a full list of command-line arguments, please check out `argument_parser.py`. Below
are some examples of running different configurations:

##### Run NMER with SAC in `InvertedPendulum-v2`: 
```
python3 custom_training.py --trainer SAC --env InvertedPendulum-v2 --custom_replay_buffer True --knn_baseline
```

##### Run Vanilla Experience Replay with SAC in `HalfCheetah-v2`: 
```
python3 custom_training.py --trainer SAC --env HalfCheetah-v2 --custom_replay_buffer False
```
##### Run Prioritized Experience Replay with DDPG in `InvertedPendulum-v2`: 
```
python3 custom_training.py --trainer DDPG --env InvertedPendulum-v2 --custom_replay_buffer False --prioritized_replay
```

##### Run GPR + Prioritized Experience Replay with SAC in `Walker2d-v2`: 
```
python3 custom_training.py --trainer SAC --env Walker2d-v2 --custom_replay_buffer True --prioritized_replay --gaussian_process --gpytorch --mc_hyper --mean_type zero --kernel rbf --use_ard --use_delta --normalize --global_hyperparams
```

Additionally, to run multiple experiments concurrently (this may be desirable if running
on single nodes in a SLURM-based cluster), this can be done by running `cluster_run.py`.

## Benchmarking, Ray RLlib, and Gaussian Process Regression Examples
To view examples of using `gpytorch` with `ray.rllib`, you can find `.ipynb` 
notebooks contained within the `examples/` directory.

## Deployment
This module has scripts designed for SLURM deployment on the MIT Lincoln Lab Supercomputing 
Center (LLSC), known as `supercloud`. If you choose to deploy to Supercloud 
(or, for that matter, any other server, please see the section below 
for some recommendations on how to use `ray` in a SLURM scheduler environment.)

### Running `mujoco-py` on MIT Supercloud:
MIT Supercloud's `gridsan` does not implement locking, and for this reason, `mujoco-py` must be 
setup from source in the storage component of this cluster. For more information
on this, please see [this GitHub issue](https://github.com/openai/mujoco-py/issues/486),
which specifically references how to solve this problem for MIT Supercloud. 
Including similar setup instructions in your bash submission scripts will ensure 
you do not run into file locking errors during setup or import.

If you would like to see the submission scripts used for deploying experiments to MIT Supercloud, 
please look at the following files:
1. `cluster_deployment/concurrent_deployment/run.sh` (Runs each experiment concurrently)
2. `cluster_deployment/single_process_deployment/run.sh` (Runs each experiment on a single process)


### Running Ray on MIT Supercloud:
- When calling `ray.init()`, which initializes a ray server, make sure to spread out 
these calls to ensure that two experiments do not start a ray server on the same port.
- Run `ray.init()` in local mode to ensure that it is run on only a single process.
- Explicitly set the number of CPUs with `num_cpus` in `ray.init()`. Currently, this
defaults to 10 CPUs.
- This allowed for deploying on nodes that have more than one experiment already running, 
as well as multiple ray experiments per node!
- Particularly for CPU-only nodes, it is not recommended to run more than 4 jobs/node.
The easiest way to do this is to simply write a different `inputs.txt` file for
each set of 4 jobs to run.
- **NOTE:** If you run `ray.init()` in `local_mode`, and you want to use GPUs, make sure 
to set the environment variable `CUDA_VISIBLE_DEVICES`. I do this when I ran my script 
calling `ray.init()` and `ray.tune()`:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
```

## Paper
Please find our 2022 L4DC **[paper](https://proceedings.mlr.press/v168/sander22a/sander22a.pdf)**, as well as **[thesis work](https://dspace.mit.edu/bitstream/handle/1721.1/138972/Sander-rmsander-meng-eecs-2021-thesis.pdf)** covering this topic.

If you find NMER useful, please consider citing our paper as:

```
@InProceedings{pmlr-v168-sander22a,
  title = 	 {Neighborhood Mixup Experience Replay: Local Convex Interpolation for Improved Sample Efficiency in Continuous Control Tasks},
  author =       {Sander, Ryan and Schwarting, Wilko and Seyde, Tim and Gilitschenski, Igor and Karaman, Sertac and Rus, Daniela},
  booktitle = 	 {Proceedings of The 4th Annual Learning for Dynamics and Control Conference},
  pages = 	 {954--967},
  year = 	 {2022},
  editor = 	 {Firoozi, Roya and Mehr, Negar and Yel, Esen and Antonova, Rika and Bohg, Jeannette and Schwager, Mac and Kochenderfer, Mykel},
  volume = 	 {168},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--24 Jun},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v168/sander22a/sander22a.pdf},
  url = 	 {https://proceedings.mlr.press/v168/sander22a.html}
}
```

## Acknowledgements
This research was supported by the Toyota Research Institute (TRI). This article solely reflects the opinions and conclusions of its authors and not TRI,
Toyota, or any other entity. We thank TRI for their support. The authors thank the MIT SuperCloud and Lincoln Laboratory Supercomputing Center for providing HPC and consultation resources that have contributed to the research results reported within this publication. 
