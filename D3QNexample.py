import gym

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from CS698R_DRLagents import D3QN
from CS698R_DRLagents.exploration_strategies import decayWrapper, selectEpsilonGreedyAction, selectGreedyAction


# make a gym environment
env = gym.make('CartPole-v0')


# pick a suitable device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# create the dueling network
class net(nn.Module):
    def __init__(self, inDim, outDim, hDim, activation = F.relu):
        super(net, self).__init__()
        self.inputlayer = nn.Linear(inDim, hDim[0])
        self.hiddenlayers = nn.ModuleList([nn.Linear(hDim[i], hDim[i+1]) for i in range(len(hDim)-1)])
        self.valuelayer = nn.Linear(hDim[-1], 1)
        self.actionadv = nn.Linear(hDim[-1], outDim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        x = self.activation(self.inputlayer(x))
        for layer in self.hiddenlayers:
            x = self.activation(layer(x))
        advantages = self.actionadv(x)
        values = self.valuelayer(x)
        qvals = values + (advantages - advantages.mean())
        return qvals
        
dueling_network = net(inDim=4, outDim=2, hDim=[8,8], activation=F.relu).to(device)


# create the exploration and exploitation strategies
explorationStrategyTrain = decayWrapper(selectEpsilonGreedyAction, 0.5, 0.05, 500, device=device)
D3QNagent = D3QN(dueling_network, env, 1, 0.8, 0.1, 10, 10000, 512, optim.RMSprop, 0.001, 500, 1, explorationStrategyTrain, selectGreedyAction, 1, device)


# train the agent and evaluate
train_stats = D3QNagent.trainAgent()
eval_rewards = D3QNagent.evaluateAgent()


# see the agent in action
for i_episode in range(5):
    observation = env.reset()
    for t in range(600):
        env.render()
        action = selectGreedyAction(D3QNagent.dueling_network, torch.tensor([observation], dtype=torch.float32, device=device))
        observation, reward, done, info = env.step(action.item())
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()