import inspect
import time
import torch
from .replaybuffer import ReplayBuffer

# Neural Fitted Q
class NFQ():
    def __init__(self, Qnetwork, env, seed, gamma, epochs, 
                 bufferSize,
                 batchSize,
                 optimizerFn,
                 optimizerLR,
                 MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES,
                 explorationStrategyTrainFn, 
                 explorationStrategyEvalFn,
                 device = torch.device('cpu'),
                 stateFn= lambda observation,info: observation, # function of observation and info, returns a state (so you can use info from .step())
                 printFrequency=50 
                ):
        
        # NOTE: explorationStrategyTrainFn and explorationStrategyEvalFn take the arguments net, state, episode(optional) in that order
        # 1. creates and initializes (with seed) the environment, train/eval episodes, gamma, etc. 
        # 2. creates and intializes all the variables required for book-keeping values via the initBookKeeping method
        # 3. creates and initializes (with network params) the optimizer function
        # 4. sets the explorationStartegy variables/functions for train and evaluation
        # 5. sets the batchSize for the number of experiences 
        # 6. Creates the replayBuffer

        # store the given params
        self.gamma, self.epochs, self.bufferSize, self.batchSize = gamma, epochs, bufferSize, batchSize
        self.MAX_TRAIN_EPISODES, self.MAX_EVAL_EPISODES = MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES
        self.device = device
        self.printFrequency = printFrequency
        self.stateFn = stateFn

        # init the book keeping
        self.initBookKeeping()

        # init the env with seed and get initial state
        env.seed(seed)
        self.initial_state = env.reset() # reset the given gym env
        self.env = env

        # the Q--neteork
        self.Qnetwork = Qnetwork

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
        self.optimizerFn = optimizerFn(self.Qnetwork.parameters(), lr=optimizerLR)

        # init the reply buffer
        self.rBuffer = ReplayBuffer(bufferSize, bufferType = 'NFQ')

        # init miscellaneous params
        self.trainreward = self.totalSteps = self.cputime = -float('inf')
        self.evalreward = self.wallclocktime = self.episode = -float('inf')
        self.totalSteps = 0



    def initBookKeeping(self):
        #this method creates and intializes all the variables required for book-keeping values and it is called in init method
        self.trainRewardsList = []
        self.evalRewardsList = []
        self.totalStepsBook = []
        self.trainTimeList = []
        self.wallClockTimeList = []



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



    def runNFQ(self):
        #this is the main method, it trains the agent, performs bookkeeping while training and finally evaluates
        #the agent and returns the following quantities:
        #1. episode wise mean train rewards
        #2. epsiode wise mean eval rewards 
        #2. episode wise trainTime (in seconds): time elapsed during training since the start of the first episode 
        #3. episode wise wallClockTime (in seconds): actual time elapsed since the start of training, 
        #                               note this will include time for BookKeeping and evaluation 
        # Note both trainTime and wallClockTime get accumulated as episodes proceed. 
        
        self.trainAgent()
        finalEvalReward = self.evalRewardsList[-1]
        
        return self.trainRewardsList, self.trainTimeList, self.evalRewardsList, self.wallClockTimeList, finalEvalReward 


    def trainAgent(self):
        #this method collects experiences and trains the NFQ agent and does BookKeeping while training. 
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
                                            countExperiences=None, net = self.Qnetwork, device= self.device)
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
            
            # uncomment to resample in every epoch
            # experiences = self.rBuffer.sample(self.batchSize)
            # states, actions, rewards, nextStates, dones = self.rBuffer.splitExperiences(experiences, device=self.device)
            
            max_a_Q = self.Qnetwork(nextStates).max(1)[0].detach()
            Q = self.Qnetwork(states).gather(1, actions)
            td_target = (rewards + self.gamma*max_a_Q*(1-dones)).unsqueeze(1)
            td_error = td_target - Q

            # loss = huberLoss(td_error).mean()
            loss = (0.5*td_error**2).mean()
            self.optimizerFn.zero_grad()
            loss.backward()
            for param in self.Qnetwork.parameters(): param.grad.clamp_(-1,1)
            self.optimizerFn.step()



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
                action = self.explorationStrategyEvalFn(self.Qnetwork, torch.tensor([state], dtype=torch.float32, device=self.device))
                observation, reward, done, info = self.env.step(action.item())
                total_reward += reward
                if render: self.env.render()
            finalEvalRewardsList.append(total_reward)
        
        self.evalreward = total_reward/self.MAX_EVAL_EPISODES   
        return finalEvalRewardsList  