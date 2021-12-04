# Deep-RL
A repo for my deep RL agents!

## Agents
The agents implemented here are:
1. NFQ
2. DQN
3. Double DQN (DDQN)
4. Dueling DDQN (D3QN)
5. Dueling DDQN with Prioritized Experience Replay (D3QN_PER)

## Other utils
1. PER Buffer
More over there is the replay buffer, which implements the Usual experience replay as well as the Prioritized Experience Replay (PER)

2. Exploration Strategies:
    - Greedy Exploration
    - Epsilon-Greedy Exploration
    - Softmax Exploration

3. Decay Wrapper:
    - Allows to decay the epsion (in case of epsilon-greedy strategy) 
    - or the temperature (in case of softmax strategy) parameters.

## Examples:
There are files with the name structure as <agent type>example.py, these are the examples of using the package for each type of agent.

To know more about the inputs (and the documentation) please read the class descriptions.
The documentation is comming in the Readme in a while...
