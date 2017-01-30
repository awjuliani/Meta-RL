# Meta-RL
Tensorflow implementation of Meta-RL A3C algorithm taken from [Learning to Reinforcement Learn](https://arxiv.org/abs/1611.05763). 
For more information, as well as explainations of each of the experiments, see my corresponding [Medium post](https://medium.com/p/b15b592a2ddf). A3C is built from previous implementation available [here](https://github.com/awjuliani/DeepRL-Agents).

Contains iPython notebooks for:

* **A3C-Meta-Bandit** - Set of bandit tasks described in paper. Including: Independent, Dependent, and Restless bandits.
* **A3C-Meta-Context** - Rainbow bandit task using randomized colors to indicate reward-giving arm in each episode. 
* **A3C-Meta-Grid** - Rainbow Gridworld task; a variation of gridworld in which goal colors are randomzied each episode and must be learned "on the fly."
