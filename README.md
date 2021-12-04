# Deep-RL
A repo for my deep RL agents!

## Agents
The agents implemented here are:
1. NFQ
2. DQN
3. Double DQN (DDQN)
4. Dueling DDQN (D3QN)
5. Dueling DDQN with Prioritized Experience Replay (D3QN_PER)
``` python 
from DRLagents import NFQ, DQN, DDQN, D3QN, D3QN_PER
```

## Other utils
1. Replay Buffer:
    ``` python
    from DRLagents.replaybuffer import ReplayBuffer
    ```
    - Implements both usual experience replay (as in DQN)
    - And the Prioritized Experience Replay (PER)
    - Can set Prioritized mode by inserting 'PER' in the bufferType example:
        ``` python
        ReplayBuffer(bufferSize, bufferType = 'PER-D3QN', priority_alpha=alpha, priority_beta=beta, priority_beta_rate=beta_rate)
        ```

2. Exploration Strategies: 
    ``` python
    from DRLagents.exploration_strategies import selectEpsilonGreedyAction, selectGreedyAction, selectSoftMaxAction
    ```
    - Greedy Exploration
    - Epsilon-Greedy Exploration
    - Softmax Exploration

3. Decay Wrapper:
    ``` python
    from DRLagents.exploration_strategies import decayWrapper
    ```
    - Allows to decay the epsion (in case of epsilon-greedy strategy) 
    - or the temperature (in case of softmax strategy) parameters.

## Examples:
The following code snippet shows how you train a deep network (torch's nn.Module) using this package. To see the full code read DQNexample.py
``` python
import gym
...
from DRLagents import D3QN
from DRLagents.exploration_strategies import decayWrapper, selectEpsilonGreedyAction, selectGreedyAction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v0')
explorationStrategyTrain = decayWrapper(selectEpsilonGreedyAction, 0.5, 0.05, 500, device=device)

DQNagent = DQN(Qnetwork, env, seed=0, gamma=0.8, epochs=10, bufferSize=10000, batchSize=512, 
                optimizerFn=optim.Adam, optimizerLR=0.001, MAX_TRAIN_EPISODES=800, MAX_EVAL_EPISODES=1, 
                explorationStrategyTrainFn= explorationStrategyTrain, explorationStrategyEvalFn= selectGreedyAction, 
                updateFrequency=5, device=device)
                
train_stats = DQNagent.trainAgent() # train the agent
eval_rewards = DQNagent.evaluateAgent()
```

There are files with the name structure as <agent type>example.py, these are the examples of using the package for each type of agent.

To know more about the inputs (and the documentation) please read the class descriptions.
The documentation is comming in the Readme in a while...
