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


class CNN_QNet(nn.Module):
    def __init__(self, in_chan, grid_rows, grid_cols):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, 32, kernel_size=3, padding=1, stride=1), 
            nn.ReLU(), 

            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(), 

            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(), 
        )

        input_size = 64 * grid_rows * grid_cols
        self.linear = nn.Sequential(
            nn.Linear(input_size, 512), 
            nn.ReLU(), 
            nn.Linear(512, 3)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class QTrainer:
    def __init__(self, model, target_model, lr, gamma):

        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Q-Value forward pass.
        self.model.train() 
        pred = self.model(state) # shape: (N x 3) (3 = # actions)
        
        # Target forward pass (evaluation mode).
        self.target_model.eval() 
        with torch.no_grad():
            next_pred = self.target_model(next_state) # shape: (N x 3)
            max_next_q = torch.max(next_pred, dim=1).values # shape: N
            
        # Compute target value (scalar).
        # Q_target := reward + gamma * max(Q(s', a'))
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
        self.optimizer.step()

        return loss.detach().item()
