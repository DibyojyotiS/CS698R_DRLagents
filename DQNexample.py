import gym

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


from DRLagents import DQN
from DRLagents.exploration_strategies import decayWrapper, selectEpsilonGreedyAction, selectGreedyAction

# make a gym environment
env = gym.make('CartPole-v0')


# pick a suitable device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# create the deep network
class net(nn.Module):
    def __init__(self, inDim, outDim, hDim, activation = F.relu):
        super(net, self).__init__()
        self.inputlayer = nn.Linear(inDim, hDim[0])
        self.hiddenlayers = nn.ModuleList([nn.Linear(hDim[i], hDim[i+1]) for i in range(len(hDim)-1)])
        self.outputlayer = nn.Linear(hDim[-1], outDim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        t = self.activation(self.inputlayer(x))
        for layer in self.hiddenlayers:
            t = self.activation(layer(t))
        t = self.outputlayer(t)
        return t

Qnetwork = net(inDim=4, outDim=2, hDim=[8,8], activation=F.relu).to(device)


# create the exploration and exploitation strategies
explorationStrategyTrain = decayWrapper(selectEpsilonGreedyAction, 0.5, 0.05, 500, device=device)
DQNagent = DQN(Qnetwork, env, seed=0, gamma=0.8, epochs=10, bufferSize=10000, batchSize=512, 
                optimizerFn=optim.Adam, optimizerLR=0.001, MAX_TRAIN_EPISODES=800, MAX_EVAL_EPISODES=1, 
                explorationStrategyTrainFn= explorationStrategyTrain, explorationStrategyEvalFn= selectGreedyAction, 
                updateFrequency=5, device=device)

                
# train the agent and evaluate
train_stats = DQNagent.trainAgent()
eval_rewards = DQNagent.evaluateAgent()


# see the agent in action
for i_episode in range(5):
    observation = env.reset()
    for t in range(600):
        env.render()
        action = selectGreedyAction(DQNagent.policy_network, torch.tensor([observation], dtype=torch.float32, device=device))
        observation, reward, done, info = env.step(action.item())
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

# for readme example
# DQN(Qnetwork, env, seed=0, gamma=0.8, epochs=10, bufferSize=10000, batchSize=512, 
#     optimizerFn=optim.Adam, optimizerLR=0.001, MAX_TRAIN_EPISODES=800, MAX_EVAL_EPISODES=1, 
#     explorationStrategyTrainFn= explorationStrategyTrain, explorationStrategyEvalFn= selectGreedyAction, 
#     updateFrequency=5, device=device)