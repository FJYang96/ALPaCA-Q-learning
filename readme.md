Course project for CS332 - Efficient exploration with ALPaCA-Q-learning

This project attempts to generalize the ALPaCA algorithm (Harrison et al. 2018)
to the domain of meta-reinforcement learning. The technical details is
documented in writeup.pdf.

The files in the alpaca directory are from the [ALPaCA
repo](https://github.com/StanfordASL/ALPaCA). The meta-mountain-car environment
is built upon the MountainCar-v0 environment from gym. The code in mmc.py
contains code from [openAI
gym](https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py).

The files are organized as follows:
```
.
|--agent/
|  |--alpacaQ.py:    the alpaca Q learning agent
|  |--random.py:     agent with random action
|--alpaca/
|  |--alpaca.py:     the original alpaca agent for meta-supervised learning
|  |--dataset.py:    dataset wrapper class for alpaca training
|--metamountaincar/
|  |--mmc.py:         mountain car environment with varying gravity and thrust
|  |--VI.py:          code for value iteration 
|  |--mmcDataset.py:  dataset wrapper for alpaca training
|  |--generateData.py:script for generating value functions for training
|--config.yml:        experiment configuration parameters
|--experiment.ipynb:  jupyter notebook for the experiments in the writeup
|--utility.py:        some utility functions for experiment and plotting
|--readme.md:         this file
```
