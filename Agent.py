from collections import deque
import random
import numpy as np
import torch
from Game import SnakeGame, Direction, Point, BLOCK_SIZE
from Model import Linear_QNet, QTrainer


# TO REMOVE.
#torch.serialization.add_safe_globals([Point, deque, np._core.multiarray._reconstruct, np.ndarray, np.dtype])

#The Feed Foreward Neural Network agent.
class FFNNAgent():
    
    def __init__(self, 
            max_dataset_size = 10_000,
            batch_size       = 32,
            lr               = 0.001,
            epsilon          = 0.1,
            decaying_epsilon = 0.999,
            min_epsilon      = 0.01,
            gamma            = 0.9,
            out_model_path   = './model.pth',
            out_csv_path     = None,
            device           = 'cpu',
            gui              = False,
            checkpoint_path  = None
    ):
        # Disount factor.
        self.gamma = gamma

        # Used for eps-greedy choice.
        self.epsilon = epsilon
        self.decaying_epsilon = decaying_epsilon
        self.min_epsilon = min_epsilon

        # Dataset with pairs (state, target)
        self.memory = deque(maxlen=max_dataset_size)
        # Used for long term train.
        self.batch_size = batch_size

        # Device.
        self.device = device

        # Action-value function approximator.
        self.model = self.model = Linear_QNet(11, 128, 3).to(self.device)

        # Model trainer.
        self.trainer = QTrainer(self.model, lr, self.gamma)

        # The game.
        self.game = SnakeGame(gui=gui)

        # Current record.
        self.record = 0
        self.record_replay = {'actions': [], 'foods': []}

        # Episodes counter.
        self.num_episodes = 0

        # Out path model and csv file.
        self.out_model_path = out_model_path
        self.out_csv_path = out_csv_path

        # Load checkpoint_path, if necesary.
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        # Create the output csv file.
        if self.out_csv_path is not None:
            with open(self.out_csv_path, 'w') as f:
                f.write("avgScore,mean_score_20,score,total_score\n")
            print("INFO: Stats will be stored in", self.out_csv_path)
    

    def get_state(self, game):
        # Get snake head.
        head = game.snake[0]
        tail = game.snake[-1]

        # Block size.
        block_sz = BLOCK_SIZE

        # Consider each position arount the head.
        pt_l = Point(head.x - block_sz, head.y)
        pt_r = Point(head.x + block_sz, head.y)
        pt_u = Point(head.x, head.y - block_sz)
        pt_d = Point(head.x, head.y + block_sz)
        
        # Define the current direction (only one is True).
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Construct the state vector.
        state = [
            # Danger straight
            (dir_r and game.is_collision(pt_r)) or 
            (dir_l and game.is_collision(pt_l)) or 
            (dir_u and game.is_collision(pt_u)) or 
            (dir_d and game.is_collision(pt_d)),

            # Danger right
            (dir_u and game.is_collision(pt_r)) or 
            (dir_d and game.is_collision(pt_l)) or 
            (dir_l and game.is_collision(pt_u)) or 
            (dir_r and game.is_collision(pt_d)),

            # Danger left
            (dir_d and game.is_collision(pt_r)) or 
            (dir_u and game.is_collision(pt_l)) or 
            (dir_r and game.is_collision(pt_u)) or 
            (dir_l and game.is_collision(pt_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]

        return np.array(state, dtype=int)


    def _remember(self, state, action, reward, next_state, gameover):
        self.memory.append((state, action, reward, next_state, gameover))


    def _train_short_memory(self, state, action, reward, next_state, gameover):
        # Transform into tensor.
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.float, device=self.device).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device).unsqueeze(0)
        gameover = torch.tensor(gameover, dtype=torch.bool, device=self.device).unsqueeze(0)
        self.trainer.train_step(state, action, reward, next_state, gameover)


    def _train_long_memory(self):
        # Define the batch.
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
        else:
            batch = self.memory

        states, actions, rewards, next_states, gameovers = [np.array(x) for x in zip(*batch)]

        # Transform in tensors.
        states = torch.tensor(states, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float, device=self.device)
        gameovers = torch.tensor(gameovers, dtype=torch.bool, device=self.device)

        self.trainer.train_step(states, actions, rewards, next_states, gameovers)


    def get_action(self, state):
        # The action to take.
        final_move = [0,0,0]

        # Get the 3 values Q(state, a) for each action a.
        self.model.eval()
        with torch.no_grad():
            state0 = torch.tensor(state, dtype=torch.float)
            state0 = torch.unsqueeze(state0, 0).to(self.device)
            prediction = self.model(state0)

        # Gready choice.
        greedy_idx = torch.argmax(prediction).item()
        # Non-gredy choices.
        rnd_actions = np.delete(np.array([0, 1, 2]), greedy_idx).tolist()
        
        # Choose the move.
        if random.random() < self.epsilon:
            # Exploration.
            action = np.random.choice(rnd_actions, p=[0.5, 0.5])
            final_move[action] = 1
        else: # Exploitation.
            final_move[greedy_idx] = 1

        return final_move


    def print_info(self, score, mean_score, mean_score_20, total_score):
        print(
            f"INFO: GAME: {self.num_episodes}\n"
            f"\tRecord: {self.record}\n"
            f"\tScore: {score}\n"
            f"\tMean score: {mean_score}\n"
            f"\tMean score last 20: {mean_score_20}\n"
            f"\tTotal score: {total_score}\n"
            f"\tepsilon: {self.epsilon}"
        )


    def _save_checkpoint(self):
        checkpoint = {
            'episode': self.num_episodes,
            'record': self.record,
            'epsilon': self.epsilon,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'memory': self.memory,
            'record_replay': self.record_replay
            # Add any other required variables (like epsilon value, etc.)
        }
        torch.save(checkpoint, self.out_model_path)


    def _load_checkpoint(self, checkpoint_path):
        # Load the checkpoint file (to CPU).
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # Apply saved states.
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore metadata.
        self.num_episodes = checkpoint['episode']
        self.record = checkpoint['record']
        self.epsilon = checkpoint['epsilon']
        self.record_replay = checkpoint['record_replay']
        self.memory = checkpoint['memory']
        print(f"INFO: Checkpoint restored. Resuming training from Episode {self.num_episodes}.")
    

    def train(self):
        total_score = 0
        score_20 = deque(maxlen=20) # Score of last 20 games.

        # Store info to replay the record.
        actions_replay = []
        food_replay = [self.game.food]

        while True:
            # Save current state.
            state_old = self.get_state(self.game)
    
            # Choose action to perform.
            final_move = self.get_action(state_old)
 
            # Do the action, save reward, gameover and current score.
            reward, gameover, score = self.game.play_step(final_move)
            state_new = self.get_state(self.game)
    
            # Train short memory.
            self._train_short_memory(state_old, final_move, reward, state_new, gameover)
    
            # Remember.
            self._remember(state_old, final_move, reward, state_new, gameover)

            # Store data for replay.
            actions_replay.append(final_move)
            food_replay.append(self.game.food)

            if gameover:
                # Train long memory, plot result.
                self.game.reset()
                self.num_episodes += 1
                self._train_long_memory()
    
                if score > self.record:
                    # Save record.
                    self.record = score
                    self.record_replay['actions'] = actions_replay
                    self.record_replay['foods'] = food_replay

                    # Create checkpoint.
                    print("INFO: New record! Saving checkpoint...")
                    self._save_checkpoint()

                # Reset replay.
                actions_replay = []
                food_replay = []
    
                # Decaying epsilon.
                self.epsilon = max(self.epsilon * self.decaying_epsilon, self.min_epsilon)
                    
                # Update stats.
                total_score += score
                mean_score = total_score / self.num_episodes
                score_20.append(score)
                mean_score_20 = sum(score_20) / len(score_20)

                # Print info in terminal.
                self.print_info(score, mean_score, mean_score_20, total_score)

                # Save stats in csv.
                if self.out_csv_path is not None:
                    with open(self.out_csv_path, 'a') as f:
                        f.write(f"{mean_score},{mean_score_20},{score},{total_score}\n")
