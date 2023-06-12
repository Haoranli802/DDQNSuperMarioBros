import torch
import torch.nn as nn
import random
import os
from network import Net

'''

equation：
    DDQN: Q*(S, A) <- r + gamma * max_a Q_target(S', a)
    DQN:  Q*(S, A) <- r + gamma * max_a Q(S', a)
'''

class DDQNAgent:
    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr,copy_step,
                 exploration_max, exploration_min, exploration_decay, test,test_models_dir):
        '''
          parameters of DDQNAgent：
              1.state_space：it's the game state space, after environment processing, the current shape is (4, 84,84)
              2.action_space：It refers to the number of actions that Mario can take
              3.max_memory_size：the size of memory used for learning
              4.batch_size：the size of data we retrieve from memory
              5.gamma：it states how much we care about the future
              6.lr：learning rate
              7 copy_step: copy the weight from local_net to target_net after the number of copy_step
              8.exploration_max：the maximum probability of taking random action
              9.exploration_min：the minimum probability of taking random action
              10.exploration_decay：exploration = exploration * exploration_decay
              11.test：to decide if the current mode is test or train
              12.test_models_dir：the directory of saving the model
        '''
        self.state_space = state_space
        self.action_space = action_space

        self.test = test

        # using gpu if we have,otherwise using cpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.local_net = Net(state_space, action_space).to(self.device)
        self.target_net = Net(state_space, action_space).to(self.device)
        self.step = 0
        self.exploration_rate = exploration_max

        if self.test:
            self.local_net.load_state_dict(torch.load(os.path.join(test_models_dir,"local_net.pt"), map_location=torch.device(self.device)))
            self.target_net.load_state_dict(torch.load(os.path.join(test_models_dir,"target_net.pt"), map_location=torch.device(self.device)))
            return

        # using Adam as optimizer
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)

        self.copy = copy_step


        # create memory used for learning

        self.max_memory_size = max_memory_size

        # initialize the learning memory
        self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
        self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
        self.ACTION_MEM = torch.zeros(max_memory_size, 1)
        self.REWARD_MEM = torch.zeros(max_memory_size, 1)
        self.DONE_MEM = torch.zeros(max_memory_size, 1)
        self.ending_position = 0
        self.num_in_queue = 0


        self.memory_sample_size = batch_size
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device)

        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    # store the new values to the memory
    def remember(self, state, action, reward, state2, done):

        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    # to get the batch size of experiences from the memory
    def get_batch_experiences(self):
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]
        return STATE, ACTION, REWARD, STATE2, DONE

    def act(self, state):
        self.step += 1
        # take the random move if the random value is less than our exp_rate
        if random.random() < self.exploration_rate:
            return torch.tensor([[random.randrange(self.action_space)]])
        return torch.argmax(self.local_net(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()

    # copy the weights from local network to our target network
    def copy_model(self):
        self.target_net.load_state_dict(self.local_net.state_dict())

    def experience_replay(self):
        if self.step % self.copy == 0:
            self.copy_model()

        if self.memory_sample_size > self.num_in_queue:
            return
        STATE, ACTION, REWARD, STATE2, DONE = self.get_batch_experiences()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)
        self.optimizer.zero_grad()
        target = REWARD + torch.mul((self.gamma * self.target_net(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)
        current = self.local_net(STATE).gather(1, ACTION.long())  # Local net approximation of Q-value
        loss = self.l1(current, target)
        loss.backward()
        self.optimizer.step()
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)