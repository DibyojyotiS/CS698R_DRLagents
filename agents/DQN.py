import inspect
import time
import torch
import copy
import numpy as np

from .replaybuffer import ReplayBuffer

class DQN():
    def __init__(self, policy_network, env, seed, gamma, epochs,
                 bufferSize,
                 batchSize,
                 optimizerFn,
                 optimizerLR,
                 MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES,
                 explorationStrategyTrainFn, 
                 explorationStrategyEvalFn,
                 updateFrequency, device= torch.device('cpu')):
        #this DQN method 
        # NOTE: explorationStrategyTrainFn and explorationStrategyEvalFn take the arguments net, state, episode(optional) in that order
        # 1. creates and initializes (with seed) the environment, train/eval episodes, gamma, etc. 
        # 2. creates and intializes all the variables required for book-keeping values via the initBookKeeping method
        # 3. creates traget network as a clone of the policy_network
        # 4. creates and initializes (with network params) the optimizer function
        # 5. sets the explorationStartegy variables/functions for train and evaluation
        # 6. sets the batchSize for the number of experiences 
        # 7. Creates the replayBuffer
    
        # store the given params
        self.gamma, self.bufferSize, self.batchSize = gamma, bufferSize, batchSize
        self.MAX_TRAIN_EPISODES, self.MAX_EVAL_EPISODES = MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES
        self.device= device
        self.updateFrequency = updateFrequency
        self.epochs = epochs

        # init the book keeping
        self.initBookKeeping()

        # init the env with seed and get initial state
        env.seed(seed)
        self.initial_state = env.reset() # reset the given gym env
        self.env = env

        # the Q-neteork and create the target network
        self.policy_network = policy_network
        self.target_network = copy.deepcopy(policy_network)

        # set the exploration strategies
        def episodeWrapper(func):
            if 'episode' in inspect.getfullargspec(func).args:
                def inner(net, state):
                    return func(net, state, episode= self.episode)
                inner.__name__ = 'episodeWrapped_' + func.__name__
                return inner
            return func
                
        self.explorationStrategyTrainFn = episodeWrapper(explorationStrategyTrainFn)
        self.explorationStrategyEvalFn = episodeWrapper(explorationStrategyEvalFn)

        # init the optmizer
        self.optimizerFn = optimizerFn(self.policy_network.parameters(), lr=optimizerLR)

        # init the reply buffer
        self.rBuffer = ReplayBuffer(bufferSize, bufferType = 'DQN')

        # init miscellaneous params
        self.trainreward = self.totalSteps = self.cputime = -float('inf')
        self.evalreward = self.wallclocktime = self.episode = -float('inf')
        self.totalSteps = 0


    def initBookKeeping(self):
        #this method creates and intializes all the variables required for book-keeping values and it is called
        self.trainRewardsList = []
        self.evalRewardsList = []
        self.totalStepsBook = []
        self.trainTimeList = []
        self.wallClockTimeList = []
        return


    def performBookKeeping(self, train = True):
        #this method updates relevant variables for the bookKeeping, this can be called 
        if train:
            self.trainRewardsList.append(self.trainreward)
            self.totalStepsBook.append(self.totalSteps)
            self.trainTimeList.append(self.cputime)
            self.wallClockTimeList.append(self.wallclocktime)
            if self.episode%50 == 0: print(f"episode: {self.episode} reward: {self.trainreward}")
        else:
            self.evalRewardsList.append(self.evalreward)
        return


    def runDQN(self):
        #this is the main method, it trains the agent, performs bookkeeping while training and finally evaluates
        #the agent and returns the following quantities:
        #1. episode wise mean train rewards
        #2. epsiode wise mean eval rewards 
        #2. episode wise trainTime (in seconds): time elapsed during training since the start of the first episode 
        #3. episode wise wallClockTime (in seconds): actual time elapsed since the start of training, 
        #                               note this will include time for BookKeeping and evaluation 

        self.trainAgent()
        finalEvalReward = np.mean(self.evaluateAgent())
        
        return self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList, finalEvalReward 


    def trainAgent(self):
        #this method collects experiences and trains the agent and does BookKeeping while training. 
        #this calls the trainNetwork() method internally, it also evaluates the agent per episode
        #it trains the agent for MAX_TRAIN_EPISODES

        # clock start times for training
        train_tcpu_start = time.time()
        train_clock_start = time.perf_counter()

        for episode in range(self.MAX_TRAIN_EPISODES):
            
            self.episode = episode
            state = self.env.reset()

            self.rBuffer.collectExperiences(self.env, state, self.explorationStrategyTrainFn, None, net = self.policy_network, device=self.device)
            if self.rBuffer.length() < self.batchSize: continue # spik episode if number of samples is less than batch size

            experiences = self.rBuffer.sample(self.batchSize)
            self.trainNetwork(experiences, self.epochs)
            
            # book keeping variables
            self.cputime = time.time() - train_tcpu_start
            self.wallclocktime = time.perf_counter() - train_clock_start
            self.trainreward = self.rBuffer.episode_reward
            self.totalSteps += self.rBuffer.episode_steps

            self.performBookKeeping(train=True)
            
            self.evaluateAgent()
            self.performBookKeeping(train=False)

            if episode%self.updateFrequency == 0:
                self.updateNetwork(self.policy_network, self.target_network)

        return self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList


    def trainNetwork(self, experiences, epochs):
        # this method trains the value network epoch number of times and is called by the trainAgent function
        # it essentially uses the experiences to calculate target, using the targets it calculates the error, which
        # is further used for calulating the loss. It then uses the optimizer over the loss 
        # to update the params of the network by backpropagating through the network
        # this function does not return anything
        # you can try out other loss functions other than MSE like Huber loss, MAE, etc. 

        states, actions, rewards, nextStates, dones = self.rBuffer.splitExperiences(experiences, device=self.device)
        for epoch in range(epochs):
            # experiences = self.rBuffer.sample(self.batchSize)
            # states, actions, rewards, nextStates, dones = self.rBuffer.splitExperiences(experiences)
            max_a_Q = self.target_network(nextStates).detach().max(1)[0]
            Q = self.policy_network(states).gather(1, actions)
            td_target = (rewards + self.gamma*max_a_Q*(1-dones)).unsqueeze(1)
            td_error = td_target - Q
            # loss = huberLoss(td_error).mean()
            loss = (0.5*td_error**2).mean()
            self.optimizerFn.zero_grad()
            loss.backward()
            for param in self.policy_network.parameters(): param.grad.clamp_(-1,1)
            self.optimizerFn.step()
        return


    def updateNetwork(self, onlineNet, targetNet):
        #this function updates the target network with the onlineNetwork
        targetNet.load_state_dict(onlineNet.state_dict())


    def evaluateAgent(self):
        #this function evaluates the agent using the value network, it evaluates agent for MAX_EVAL_EPISODES
        #typcially MAX_EVAL_EPISODES = 1

        finalEvalRewardsList = []
        for evalepisode in range(self.MAX_EVAL_EPISODES):
            state = self.env.reset()
            done = False
            total_reward = 0
            self.episode = evalepisode # self.episode will be overwritten in training loop ok to use here
            while not done:
                action = self.explorationStrategyEvalFn(self.policy_network, torch.tensor([state], dtype=torch.float32, device=self.device))
                state, reward, done, _ = self.env.step(action.item())
                total_reward += reward
            finalEvalRewardsList.append(total_reward)
        
        self.evalreward = total_reward/self.MAX_EVAL_EPISODES
        
        return finalEvalRewardsList  