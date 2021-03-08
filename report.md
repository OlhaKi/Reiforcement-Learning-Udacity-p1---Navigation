#### Project 1 : Navigation
## Examine the State and Action Spaces

The simulation contains a single agent that navigates a large environment. At each time step, it has four actions at its disposal:

    0 - walk forward
    1 - walk backward
    2 - turn left
    3 - turn right

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.

The goal of our agent is to collect as many yellow bananas as possible while avoiding blue bananas. In order to solve the environment, our agent must achieve an average score of +13 over 100 consecutive episodes.

My project  is adapting  the code from the exercise(Deep Q-Networks lesson)  to the project.
The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.
## Learning algorithm - Deep Q-Networks
## Chosen hyperparameters:
    BUFFER_SIZE   100000
    BATCH_SIZE   64
    GAMMA   0.99
    TAU   0.001
    LR   0.0005
    UPDATE_EVERY   4

## Model architectures for neural network:
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=296, fc2_units=296):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
## Performance of the agent
   # Plot of Rewards
   ![GitHub Logo](/Images/p1_1.png)
   # Testing my model:
   ![GitHub Logo](/Images/p1_2.png)   
The complete set of results and steps can be found in my notebook   
## Ideas for Future Work:
1. Implementation Double Deep Q Networks with Prioritized Experience Replay
2. I am planning to train DQN with pixels environment.
