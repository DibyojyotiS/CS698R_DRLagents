import inspect
import time
import torch
import copy
import numpy as np

from .replaybuffer import ReplayBuffer

# Dueling DDQN with prioritized experience replay
class D3QN_PER():
    def __init__(self, dueling_network, env, seed, gamma, tau, alpha, beta, beta_rate, epochs,
                 bufferSize,
                 batchSize,
                 optimizerFn,
                 optimizerLR,
                 MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES,
                 explorationStrategyTrainFn, 
                 explorationStrategyEvalFn,
                 updateFrequency, 
                 device= torch.device('cpu'),
                 stateFn= lambda observation,info: observation, # function of observation and info, returns a state (so you can use info from .step())
                 printFrequency=50 
                 ):
        #this D3QN_PER method 
        # 1. creates and initializes (with seed) the environment, train/eval episodes, gamma, etc. 
        # 2. creates and intializes all the variables required for book-keeping values via the initBookKeeping method
        # 3. creates traget network as a clone of the policy_network
        # 4. creates and initializes (with network params) the optimizer function
        # 5. sets the explorationStartegy variables/functions for train and evaluation
        # 6. sets the batchSize for the number of experiences 
        # 7. Creates the replayBuffer, 
        #    the replayBuffer takes the parameters bufferSize, alpha, beta and beta_rate
        
        # store the given params
        self.gamma, self.bufferSize, self.batchSize = gamma, bufferSize, batchSize
        self.MAX_TRAIN_EPISODES, self.MAX_EVAL_EPISODES = MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES
        self.updateFrequency = updateFrequency
        self.epochs = epochs
        self.device = device
        self.stateFn = stateFn
        self.printFrequency = printFrequency

        # init the book keeping
        self.initBookKeeping()

        # init the env with seed and get initial state
        env.seed(seed)
        self.initial_state = env.reset() # reset the given gym env
        self.env = env

        # create the Q--neteork
        self.dueling_network = dueling_network
        self.target_network = copy.deepcopy(dueling_network)

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
        self.optimizerFn = optimizerFn(self.dueling_network.parameters(), lr=optimizerLR)

        # init the reply buffer
        self.tau, self.alpha, self.beta, self.beta_rate = tau, alpha, beta, beta_rate
        self.rBuffer = ReplayBuffer(bufferSize, bufferType = 'PER-D3QN', priority_alpha=alpha, 
                                        priority_beta=beta, priority_beta_rate=beta_rate)

        # init miscellaneous params
        self.trainreward = self.totalSteps = self.cputime = -float('inf')
        self.evalreward = self.wallclocktime = self.episode = -float('inf')
        self.totalSteps = 0


    def initBookKeeping(self):
        #this method creates and intializes all the variables required for book-keeping values and it is called
        #init method
        self.trainRewardsList = []
        self.evalRewardsList = []
        self.totalStepsBook = []
        self.trainTimeList = []
        self.wallClockTimeList = []
        return


    def performBookKeeping(self, train = True):
        #this method updates relevant variables for the bookKeeping, this can be called 
        #multiple times during training
        #if you want you can print information using this, so it may help to monitor progress and also help to debug
        if train:
            self.trainRewardsList.append(self.trainreward)
            self.totalStepsBook.append(self.totalSteps)
            self.trainTimeList.append(self.cputime)
            self.wallClockTimeList.append(self.wallclocktime)
            if self.episode%self.printFrequency == 0: print(f"episode: {self.episode} reward: {self.trainreward}")
        else:
            self.evalRewardsList.append(self.evalreward)
        return 


    def runD3QN_PER(self):
        #this is the main method, it trains the agent, performs bookkeeping while training and finally evaluates
        #the agent and returns the following quantities:
        #1. episode wise mean train rewards
        #2. epsiode wise mean eval rewards 
        #2. episode wise trainTime (in seconds): time elapsed during training since the start of the first episode 
        #3. episode wise wallClockTime (in seconds): actual time elapsed since the start of training, 
        #                               note this will include time for BookKeeping and evaluation 
        # Note both trainTime and wallClockTime get accumulated as episodes proceed. 
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
            observation = self.env.reset()
            state = self.stateFn(observation, None)

            self.rBuffer.collectExperiences(self.env, state, self.stateFn, self.explorationStrategyTrainFn,
                                            countExperiences=None, net = self.dueling_network, device= self.device)

            if self.rBuffer.length() < self.batchSize: continue # skip episode if number of samples is less than batch size

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

            self.updateNetwork(self.dueling_network, self.target_network)

            if episode%self.updateFrequency == 0:
                self.target_network.load_state_dict(self.dueling_network.state_dict())
        
        return self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList


    def trainNetwork(self, experiences, epochs):
        # this method trains the value network epoch number of times and is called by the trainAgent function
        # it essentially uses the experiences to calculate target, using the targets it calculates the error, which
        # is further used for calulating the loss. It then uses the optimizer over the loss 
        # to update the params of the network by backpropagating through the network
        # this function does not return anything
        # you can try out other loss functions other than MSE like Huber loss, MAE, etc. 
        states, actions, rewards, nextStates, dones, importance_weights, sample_idx = self.rBuffer.splitExperiences(experiences, self.device)
    
        for epoch in range(epochs):

            # uncomment for resampling for each update
            # experiences = self.rBuffer.sample(self.batchSize)
            # states, actions, rewards, nextStates, dones, importance_weights, sample_idx = self.rBuffer.splitExperiences(experiences, self.device)
            
            argmax_a_Q = self.dueling_network(nextStates).max(1, keepdims=True)[1]
            max_a_Q = self.target_network(nextStates).detach().gather(1, argmax_a_Q).squeeze()

            Q = self.dueling_network(states).gather(1, actions)
            td_target = (rewards + self.gamma*max_a_Q*(1-dones)).unsqueeze(1)
            td_error = td_target - Q
            # loss = (importance_weights*huberLoss(td_error)).mean() # uncomment for huber loss
            loss = (importance_weights*(0.5*td_error**2)).mean()
            self.optimizerFn.zero_grad()
            loss.backward()
            for param in self.dueling_network.parameters(): param.grad.clamp_(-1,1)
            self.optimizerFn.step()
            
            # update priorities in buffer
            self.rBuffer.update(sample_idx, td_error)

        return


    def updateNetwork(self, onlineNet, targetNet):
        #this function updates the onlineNetwork with the target network using Polyak averaging \
        with torch.no_grad():
            for paramOnline, paramTarget in zip(onlineNet.parameters(), targetNet.parameters()):
                paramTarget.data = self.tau * paramOnline.data + (1 - self.tau) * paramTarget.data
        return


    def evaluateAgent(self, render=False):
        #this function evaluates the agent using the value network, it evaluates agent for MAX_EVAL_EPISODES
        #typcially MAX_EVAL_EPISODES = 1

        finalEvalRewardsList = []
        for evalepisode in range(self.MAX_EVAL_EPISODES):

            observation = self.env.reset()
            info = None

            done = False
            total_reward = 0
            self.episode = evalepisode # self.episode will be overwritten in training loop ok to use here
            while not done:
                state = self.stateFn(observation, info)
                action = self.explorationStrategyEvalFn(self.dueling_network, torch.tensor([state], dtype=torch.float32, device=self.device))
                observation, reward, done, info = self.env.step(action.item())
                total_reward += reward
                if render: self.env.render()
            finalEvalRewardsList.append(total_reward)
        
        self.evalreward = total_reward/self.MAX_EVAL_EPISODES
                
        return finalEvalRewardsList  