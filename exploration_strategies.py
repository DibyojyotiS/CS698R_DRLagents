import numpy as np

import torch
import torch.nn.functional as F


def selectGreedyAction(net, state, device='cpu'):
    #this function gets q-values via the network and selects greedy action from q-values and returns it
    with torch.no_grad():
        Qpred = net(state) # predicted q-function, assumed to be 2D
        greedyAction = torch.argmax(Qpred, dim=1, keepdim=True)
    return greedyAction



def selectEpsilonGreedyAction(net, state, epsilon=0.1, device='cpu'):
    #this function gets q-values via the network and selects an action from q-values using epsilon greedy strategy
    #and returns it
    #note this function can be used for decaying epsilon greedy strategy, using the wrapper decayWrapper 

    with torch.no_grad():
        Qpred = net(state)
        sample = torch.rand(1)
        if sample < epsilon:
            eGreedyAction = torch.randint(Qpred.shape[1], (Qpred.shape[0],1)).to(device)
        else:
            eGreedyAction = torch.argmax(Qpred, dim=1, keepdim=True)

    return eGreedyAction



def selectSoftMaxAction(net, state, temp=1, device='cpu'):
    #this function gets q-values via the network and selects an action from q-values using softmax strategy
    #and returns it
    #note this function can be used for decaying temperature softmax strategy, using the wrapper decayWrapper

    with torch.no_grad():
        Probs = F.softmax(net(state)/temp, dim=1)
        softAction = torch.distributions.Categorical(Probs).sample()
    return softAction.view(-1,1)



def decayWrapper(function_with_net_state_ParamToDecay_as_args, param_init, param_final, decaysteps):
    # param decays to 1/e of initial in decaysteps number of episodes

    def inner(net, state, episode):
        decayed_epsilon = param_final + (param_init-param_final)*np.exp(-1 * episode/decaysteps)
        return function_with_net_state_ParamToDecay_as_args(net, state, decayed_epsilon)
    inner.__name__ = 'decayWrapped_' + function_with_net_state_ParamToDecay_as_args.__name__
    return inner