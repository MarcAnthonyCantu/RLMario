import torch
import random, numpy as np
from pathlib import Path

from neural import MarioNet
from collections import deque


class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None, reset_epsilon=False, set_epsilon=1, epsilon_min=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.epsilon = 1 #Exploration rate
        self.epsilon_decay = 0.9999995
        self.epsilon_min = epsilon_min
        self.gamma = 0.99 #Discount factor

        self.curr_step = 0
        self.burnin = 3e4  # min. experiences before training 10000
        self.learn_every = 3   # no. of experiences between updates to Q_online
        self.sync_every = 1e4   # no. of experiences between Q_target & Q_online sync

        self.save_every = 5e4   # no. of experiences between saving Mario Net
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
            
        if checkpoint:
            self.load(checkpoint, reset_epsilon, set_epsilon)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()


    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = np.array(state)
            state = torch.from_numpy(state).float() / 225.0
            state = state.cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_reward = self.net(state, model='online') #Outputs reward vaules for actions associated to the inputed state
            action_idx = torch.argmax(action_reward, axis=1).item()

        # decrease exploration_rate
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer).
        Stored on CPU as uint8 to minimize memory; converted to float on sample.
        """
        # Convert LazyFrames -> numpy if needed
        if hasattr(state, '__array__'):
            state = np.array(state)
            next_state = np.array(next_state)
    
        # Store on CPU. Use uint8 if frames are 0-255 (standard for Mario wrappers).
        # If your wrapper already normalizes to 0-1 float, change dtype to torch.float32
        # and remove the /255.0 in recall().
        state = torch.from_numpy(state).to(torch.uint8)
        next_state = torch.from_numpy(next_state).to(torch.uint8)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)

        self.memory.append((state, next_state, action, reward, done))


    def recall(self):
        """
        Retrieve a batch from memory and move it to GPU.
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
    
        if self.use_cuda:
            # Non-blocking transfer; normalize uint8 -> float32 on GPU
            state = state.to('cuda', non_blocking=True).float() / 255.0
            next_state = next_state.to('cuda', non_blocking=True).float() / 255.0
            action = action.to('cuda', non_blocking=True)
            reward = reward.to('cuda', non_blocking=True)
            done = done.to('cuda', non_blocking=True)
        else:
            state = state.float() / 255.0
            next_state = next_state.float() / 255.0
    
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action] # Q_online(s,a)
        return current_Q


    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    def update_Q_online(self, td_estimate, td_target) :
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.epsilon
            ),
            save_path
        )
        #print(f"MarioNet saved to {save_path} at step {self.curr_step}")


    def load(self, load_path, reset_epsilon, set_epsilon):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        epsilon = ckp.get('exploration_rate')
        state_dict = ckp.get('model')
        self.net.load_state_dict(state_dict)

        if not reset_epsilon:
            self.epsilon = epsilon
        else:
            self.epsilon = set_epsilon
        print(f"Loading model at {load_path} with exploration rate {self.epsilon}")
