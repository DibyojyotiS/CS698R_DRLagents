from collections import deque
import numpy as np
import random
import torch

class ReplayBuffer():
    def __init__(self, bufferSize, bufferType = 'DQN', **kwargs):
        # this function creates the relevant data-structures, and intializes all relevant variables
        # it can take variable number of parameters like alpha, beta, beta_rate (required for PER)
        # here the bufferType variable can be used to maintain one class for all types of agents
        # using the bufferType parameter in the methods below, you can implement all possible functionalities 
        # that could be used for different types of agents
        # permissible values for bufferType = NFQ, DQN, DDQN, D3QN and PER-D3QN
        
        assert bufferType in ['NFQ', 'DQN', 'DDQN', 'D3QN' ,'PER-D3QN']
        self.bufferType = bufferType
        self.buffer = deque(maxlen=bufferSize)
        self.isPriorityBuffer = False
        if bufferType[0:3] == 'PER':
            self.isPriorityBuffer = True
            self.priorities = deque(maxlen=bufferSize)
            self.priority_alpha = kwargs['priority_alpha']
            self.priority_beta  = kwargs['priority_beta']
            self.priority_beta_rate = kwargs['priority_beta_rate']

        # miscellaneous params
        self.episode_reward = self.episode_steps = 0
        return # just adding a return statement for for readablity



    def store(self, experience):
        #stores the experiences, based on parameters in init it can assign priorities, etc.  
        #this function does not return anything
        self.buffer.append(experience)
        if self.isPriorityBuffer: self.priorities.append(max(self.priorities, default=1))
    


    def collectExperiences(self, env, state, stateFn, explorationStrategy, countExperiences, net, device):
        #this method allows the agent to interact with the environment starting from a state and it collects
        #experiences during the interaction, it uses network to get the value function and uses exploration strategy
        #to select action. It collects countExperiences and in case the environment terminates before that it returns
        #the function calling this method handles early termination accordingly.
        #this function does not return anything
        
        experiences = []
        cumilative_reward = 0
        steps = 0
        done = False

        # run the entire episode to find the total steps
        while not done:
            steps += 1
            action = explorationStrategy(net, torch.tensor([state], dtype=torch.float32, device=device))
            newObservation, reward, done, info = env.step(action.item())
            newState = stateFn(newObservation, info) # get the state from a user defined combination of observation and info

            # collect only the first countExperiences as asked
            if countExperiences is None or steps <= countExperiences:
                experiences.append([torch.tensor([state], dtype=torch.float32, device=device),
                                    action, # Long type tensor
                                    torch.tensor([reward], dtype=torch.float32, device=device),
                                    torch.tensor([newState], dtype=torch.float32, device=device),
                                    torch.tensor([done], dtype=torch.int, device=device)    ])

            cumilative_reward += reward
            if done: break
            state = newState

        self.episode_reward = cumilative_reward
        self.episode_steps = steps

        # increment the beta with every episode
        if self.isPriorityBuffer: self.priority_beta = min(1, self.priority_beta + self.priority_beta_rate) 
        
        if countExperiences is not None and len(experiences) < countExperiences: return
        for experience in experiences:
            self.store(experience)



    def update(self, indices, newpriorities):
        #this is mainly used for PER-DDQN
        #otherwise just have a pass in this method
        #this function does not return anything
        if not self.isPriorityBuffer: return
        newpriorities = torch.abs(newpriorities)
        for idx, newpriority in zip(indices, newpriorities):
            self.priorities[idx] = newpriority.item() + 0.000000001



    def sample(self, batchSize, **kwargs):
        # kwargs: sample_type. If sample_type is 'latest' then batchSize amount of most recent experiences are returned
        # this method returns batchSize number of experiences
        # based on extra arguments, it could do sampling or it could return the latest batchSize experiences or via some other strategy
        # in the case of Prioritized Experience Replay (PER) the sampling takes into account the priorities
        # this function returns experiences samples
        # in case of Prioritized Experience Replay (PER) this function returns a list of experiences samples with their corresponding 
        # importance_weights, sample_idx as a single list of length batchSize 

        if 'sample_type' in kwargs:
            sample_type = kwargs['sample_type']
            if sample_type == 'latest':
                return self.buffer[-batchSize:] # last batchSize experiences

        if self.bufferType[:3] == 'PER':
            # prioritized sampling
            scaled_priorities = np.array(self.priorities)**self.priority_alpha
            sample_probablities = scaled_priorities/np.sum(scaled_priorities)
            sample_idx = random.choices(range(len(self.buffer)), k=batchSize, weights=sample_probablities)

            experiencesList = np.array(self.buffer)[sample_idx]

            weights = (1/len(self.buffer))*(1/sample_probablities[sample_idx])
            weights = weights ** self.priority_beta
            importance_weights = weights/np.max(weights)

            # append the importance_weights, sample_idx to the experiencesList
            combinedList = [*zip(experiencesList, importance_weights, sample_idx)]
            return combinedList
            # return experiencesList, importance_weights, sample_idx
        else:
            # uniform sampling
            experiencesList = random.sample(self.buffer, batchSize)
            return experiencesList



    def splitExperiences(self, experiences, device):
        # it takes in experiences and gives the following:
        # states, actions, rewards, nextStates, dones
        # in case of Prioritized Experience Replay (PER) it returns:
        # states, actions, rewards, nextStates, dones, importance_weights, sample_idx

        if self.isPriorityBuffer:
            experiences, importance_weights, sample_idx  = [*zip(*experiences)]

        experiencesList = [*zip(*experiences)]
        states = torch.cat(experiencesList[0]).to(device) 
        actions = torch.cat(experiencesList[1]).to(device) 
        rewards = torch.cat(experiencesList[2]).to(device) 
        nextStates = torch.cat(experiencesList[3]).to(device) 
        dones = torch.cat(experiencesList[4]).to(device) 

        if self.isPriorityBuffer:
            importance_weights = torch.tensor([importance_weights], device=device).view(-1,1)
            return states, actions, rewards, nextStates, dones, importance_weights, sample_idx

        return states, actions, rewards, nextStates, dones



    def length(self):
        #tells the number of experiences stored in the internal buffer
        bufferSize = len(self.buffer)
        return bufferSize
