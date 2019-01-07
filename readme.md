Course project for CS332 - Efficient exploration with ALPaCA-Q-learning

This project attempts to generalize the ALPaCA algorithm (Harrison et al. 2018)
to the domain of meta-reinforcement learning. The technical details is
documented in writeup.pdf.

The files in the alpaca directory are from the [ALPaCA
repo](https://github.com/StanfordASL/ALPaCA). The meta-mountain-car environment
is built upon the MountainCar-v0 environment from gym. The code in mmc.py
contains code from [openAI
gym](https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py).

The files are organized as following:
```
.
|--agent/
|  |--alpacaQ.py:       the alpaca Q learning agent
|  |--random.py:        agent with random action
|  |--alpaca.py:        implementation of the original alpaca algorithm
|--metamountaincar/
|  |--mmc.py:           mountain car environment with varying gravity and thrust
|  |--VI.py:            code for value iteration 
|  |--mmcDataset.py:    dataset wrapper for alpaca training
|  |--generateData.py:  script for generating value functions for training
|  |--dataset.py:       dataset wrapper class for alpaca training
|  |--utility.py:       utility functions for experiment and plotting
|--config.yml:          experiment configuration parameters
|--experiment.ipynb:    jupyter notebook for the experiments in the writeup
|--readme.md
|--writeup.pdf
```

This repository is still under construction. The issues include:
- ~~rewrite plotting functions~~
- ~~rewrite offline dataset to use the normalizer~~
- ~~move sample_parameters method into MMC class. Have MMC randomly sample a set
    of environment parameters during initialization~~
- ~~make sure saving and loading works for alpaca offline agent~~
- ~~test online prediction of alpaca offline agents~~
- ~~rewrite alpaca Q agents and separate them into different classes~~
- rerun experiments and see it they work
- bug in value iteration. (T-table has indices over bounds). If had time,
write triangle discretization to replace naive discretization.

Questions and suggestions are very welcome. You can reach me at fyang3[attt]stanford[dotttt]edu
