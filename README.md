[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

This project contains a solution to the first project of Udacity Deep Reinforcement Learning.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Environment
#### Install drlnd environment
1. Create (and activate) a new environment with Python 3.6.
```console
$ conda create --name drlnd python=3.6
$ source activate drlnd
```

2. Follow the instructions in this repository to perform a minimal install of [OpenAI gym](https://github.com/openai/gym).
```console
$ pip install gym
```

3. Clone the repository (if you haven't already!), and navigate to the python/ folder.
   Then, install several dependencies.
```console
$ git clone https://github.com/udacity/deep-reinforcement-learning.git
$ cd deep-reinforcement-learning/python
$ pip install .
```

4. Create an IPython kernel for the drlnd environment.
```console
$ python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the drlnd environment
   by using the drop-down Kernel menu.
   
   
### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  
The successfully trained DQN parameters is saved in the file `checkpoint.pth`. 


### Learning Algorithm

The agent is trained using Deep Q-Learning algorithm(see Report.md).

### My ideas for future work

This project used the basic [DQN algorithm](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). It can be improved by applying the following methods:

    double DQN
    dueling DQN
