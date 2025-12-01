import torch
import torch.nn as nn
import torch.optim as optim

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QTrainer:
    def __init__(self, model, lr, gamma, device):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
        self.device = device


    def train_step(self, state, action, reward, next_state, done):
        # Move everything to device.
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.float, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device)
    
        # Needed when training short memory.
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)
    
        # Q-Value forward pass.
        self.model.train() 
        pred = self.model(state) # shape: (N x 3) (3 = # actions)
        
        # Target forward pass (evaluation mode).
        self.model.eval() 
        with torch.no_grad():
            next_pred = self.model(next_state) # shape: (N x 3)
            max_next_q = torch.max(next_pred, dim=1).values # shape: N
            
        
        # Compute target value (scalar).
        # Q_target = reward + gamma * max(Q(s', a'))
        Q_new = reward + self.gamma * max_next_q # shape: N
        
        # Q_target is just the reward if the episode is done (terminal state).
        Q_target = torch.where(done, reward, Q_new) # shape: N
    
        # We only update the Q-value for the action 
        # that was actually taken (action[idx]).

        # Find the index of the action taken.
        # (dim=1 means argmax over the action dimension)
        action_indices = torch.argmax(action, dim=1, keepdim=True)  # shape: (N x 1)
    
        # Efficiently update only the relevant indices
        target = pred.clone().detach() # => calculation doesn't affect gradients.
        target.scatter_(1, action_indices, Q_target.unsqueeze(1))
    
        # Optimize model.
        self.model.train() 
        self.optimizer.zero_grad()
        
        # Compute loss; backpropagate.
        loss = self.criterion(target, pred)
        loss.backward()
        
        # Clip gradients to prevent exploding gradients.
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) 

        self.optimizer.step()

