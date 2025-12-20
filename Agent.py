from collections import deque
import random
import numpy as np
import torch
from Game import SnakeGame, Direction, Point, BLOCK_SIZE
from Model import *
import os
from Utils import ReplayBuffer


class BlindAgent():
    
    def __init__(self, 
            max_dataset_size = 10_000,
            batch_size       = 32,
            lr               = 0.001,
            epsilon          = 0.1,
            decaying_epsilon = 0.999,
            min_epsilon      = 0.01,
            gamma            = 0.9,
            target_sync      = 100,
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

        # Device.
        self.device = device

        # Dataset with pairs (state, target)
        self.memory = ReplayBuffer(
                state_shape=(11, ),
                action_dim=3,
                max_size=max_dataset_size,
                device=self.device
        )
        self.batch_size = batch_size

        self.fixed_states = torch.zeros((250, 11), device=self.device)

        # Action-value function approximator.
        self.model = Linear_QNet(11, 128, 3).to(self.device)
        self.target_model = Linear_QNet(11, 128, 3).to(self.device)
        self.target_sync = target_sync

        # Model trainer.
        self.trainer = QTrainer(
                self.model,
                self.target_model,
                lr,
                self.gamma
        )

        # The game.
        self.game = SnakeGame(gui=gui)

        # Current record.
        self.record = 0
        self.record_replay = {'actions': [], 'foods': []}

        # Episodes and steps counters.
        self.num_episodes = 0
        self.num_steps = 0

        # Out path model and csv file.
        self.out_model_path = out_model_path
        self.out_csv_path = out_csv_path

        # Load checkpoint_path, if necesary.
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        # Create the output csv file.
        if self.out_csv_path is not None:
            if not os.path.exists(self.out_csv_path):
                with open(self.out_csv_path, 'w') as f:
                    f.write("mean_score,mean_score_100,score,epsilon,max_loss,avg_q\n")
            print("INFO: Stats will be stored in", self.out_csv_path)
    
    
    def _target_sync(self):
        """Synchronise main network and target network parameters"""
        self.target_model.load_state_dict(self.model.state_dict())


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

        return torch.tensor(state, dtype=torch.float)


    def _remember(self, state, action, reward, next_state, gameover):
        self.memory.append(state, action, reward, next_state, gameover)


    def _train_long_memory(self):
        # Create batch.
        states, actions, rewards, next_states, gameovers = self.memory.sample_buffer(self.batch_size)

        # Train and return loss.
        loss = self.trainer.train_step(states, actions, rewards, next_states, gameovers)
        return loss 


    def get_action(self, state):
        # The action to take.
        final_move = [0,0,0]

        # Get the 3 values Q(state, a) for each action a.
        self.model.eval()
        with torch.no_grad():
            state0 = state.detach().clone()
            state0 = torch.unsqueeze(state0, 0).to(self.device)
            prediction = self.model(state0)

        # Gready choice.
        greedy_idx = torch.argmax(prediction).item()
        # Non-gready choices.
        rnd_actions = np.delete(np.array([0, 1, 2]), greedy_idx).tolist()
        
        # Choose the move.
        if self.num_steps < 1000:
            action = random.randint(0, 2)
            final_move[action] = 1
        elif random.random() < self.epsilon: # Exploration.
            action = np.random.choice(rnd_actions, p=[0.5, 0.5])
            final_move[action] = 1
        else: # Exploitation.
            final_move[greedy_idx] = 1

        return final_move


    def _save_checkpoint(self):
        checkpoint = {
            'episode': self.num_episodes,
            'record': self.record,
            'epsilon': self.epsilon,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            #'memory': self.memory,
            'record_replay': self.record_replay
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
        #self.memory = checkpoint['memory']
        print(f"INFO: Checkpoint restored.")
    

    def _target_sync(self):
        """Synchronise main network and target network parameters"""
        self.target_model.load_state_dict(self.model.state_dict())


    def train(self):
        total_score = 0
        score_last_100 = deque(maxlen=100) # Score of last 100 games.

        episode_steps = 0
        episode_max_loss = 0

        # Store info to replay the record.
        actions_replay = []
        episode_max_loss = 0
        food_replay = [self.game.food]

        while True:
            # Save current state.
            state_old = self.get_state(self.game)
            
            if self.num_steps < 250:
                self.fixed_states[self.num_steps] = state_old
    
            # Choose action to perform.
            final_move = self.get_action(state_old)
            self.num_steps += 1
            episode_steps += 1
 
            # Do the action, save reward, gameover and current score.
            reward, gameover, score = self.game.play_step(final_move)
            state_new = self.get_state(self.game)
    
            # Train step.
            if len(self.memory) > self.batch_size:
                loss = self._train_long_memory()
                episode_max_loss = max(episode_max_loss, loss)
    
            # Remember.
            self._remember(state_old, final_move, reward, state_new, gameover)

            # Store data for replay.
            actions_replay.append(final_move)
            food_replay.append(self.game.food)
            
            # Sync networks if necessary.
            if not self.num_steps % self.target_sync:
                print("Info: Syncronizing target model with main model...")
                self._target_sync()

            if gameover:
                # Train long memory, plot result.
                self.game.reset()
                self.num_episodes += 1
    
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
                score_last_100.append(score)
                mean_last_100 = sum(score_last_100) / len(score_last_100)

                self.model.eval()
                with torch.no_grad():
                    avg_q = self.model(self.fixed_states)
                    avg_q = torch.max(avg_q, dim=1).values # shape: N
                    avg_q = avg_q.mean().item()

                # Log info to terminal.
                print(
                    "INFO:\n"
                    f"\tGAME: {self.num_episodes}\n"
                    f"\tRecord: {self.record}\n"
                    f"\tSteps: {self.num_steps}\n"
                    f"\tBuffer memory size: {len(self.memory)}\n"
                    f"\tScore: {score}\n"
                    f"\tDuration (steps): {episode_steps}\n"
                    f"\tMean score: {mean_score}\n"
                    f"\tMean score last 100: {mean_last_100}\n"
                    f"\tTotal score: {total_score}\n"
                    f"\tepsilon: {self.epsilon}"
                )

                csv_line = f"{mean_score},{mean_last_100},{score},{self.epsilon},{episode_max_loss},{avg_q}\n"

                # Save stats in csv.
                if self.out_csv_path is not None:
                    with open(self.out_csv_path, 'a') as f:
                        f.write(csv_line)

                # Restore stats.
                episode_max_loss = 0
                episode_steps = 0

                # Save stats in csv.
                if self.out_csv_path is not None:
                    with open(self.out_csv_path, 'a') as f:
                        f.write(f"{mean_score},{mean_last_100},{score},{total_score}\n")


# The ATARI agent.
class AtariAgent:
    def __init__(self,
            max_dataset_size = 10_000,
            batch_size       = 32,
            lr               = 0.001,
            epsilon          = 0.1,
            min_epsilon      = 0.01,
            gamma            = 0.9,
            target_sync      = 100,
            out_model_path   = './model.pth',
            out_csv_path     = None,
            device           = 'cpu',
            gui              = False,
            checkpoint_path  = None,
            load_buffer      = False

    ):
        # Disount factor.
        self.gamma = gamma

        # Used for eps-greedy choice.
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon

        # The game.
        self.game = SnakeGame(gui=gui)
        self.frame_rows = self.game.h // BLOCK_SIZE + 2
        self.frame_cols = self.game.w // BLOCK_SIZE + 2

        # Define the frame stack.
        self.stack_size = 4
        self.frames = deque(maxlen=self.stack_size)
        self._reset_stack()

        # Dataset with pairs (state, target)
        self.memory = ReplayBuffer(
                state_shape=(self.stack_size, self.frame_rows, self.frame_cols),
                action_dim=3,
                max_size=max_dataset_size,
                device=device
        )
        self.batch_size = batch_size

        # Device.
        self.device = device

        # To monitor action value (Q)
        self.fixed_states = torch.zeros((250, 4, self.frame_rows, self.frame_cols), device=self.device)

        # Action-value function approximator (main model).
        self.model = CNN_QNet(
                in_chan=4,
                grid_rows=self.frame_rows,
                grid_cols=self.frame_cols
        ).to(self.device)

        # Target model.
        self.target_model = CNN_QNet(
                in_chan=4,
                grid_rows=self.frame_rows,
                grid_cols=self.frame_cols
        ).to(self.device)

        # Num of steps after which the networks are synchronized.
        self.target_sync = target_sync

        # Model trainer.
        self.trainer = QTrainer(
                self.model,
                self.target_model,
                lr,
                self.gamma
        )

        # Current record.
        self.record = 0
        self.record_replay = {'actions': [], 'foods': []}

        # Episodes and steps counter.
        self.num_episodes = 0
        self.num_steps = 0

        # Out path model and csv file.
        self.out_model_path = out_model_path
        self.out_csv_path = out_csv_path

        # Load checkpoint_path, if necesary.
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, load_buffer)

        # Create the output csv file.
        if self.out_csv_path is not None:
            if not os.path.exists(self.out_csv_path):
                with open(self.out_csv_path, 'w') as f:
                    f.write("mean_score,mean_score_100,score,mean_reward_100,epsilon,max_loss,avg_q\n")
            print("INFO: Stats will be stored in", self.out_csv_path)
 

    def _reset_stack(self):
        for _ in range(self.stack_size):
            zero_frame = torch.zeros((self.frame_rows, self.frame_cols), dtype=torch.float)
            self.frames.append(zero_frame)


    def _get_single_frame(self, game):
        # Initialize empty grid.
        dims = (self.frame_rows, self.frame_cols)
        state = torch.zeros(dims, dtype=torch.float)

        # Draw body (value=0.5).
        for point in game.snake[1:]:
            i = int(point.y // BLOCK_SIZE) + 1
            j = int(point.x // BLOCK_SIZE) + 1
            state[i, j] = 0.5

        # Draw head (value=1.0)
        i = int(game.head.y // BLOCK_SIZE) + 1
        j = int(game.head.x // BLOCK_SIZE) + 1
        state[i, j] = 1.0

        # Draw food (value=2.0)
        i = int(game.food.y // BLOCK_SIZE) + 1
        j = int(game.food.x // BLOCK_SIZE) + 1
        state[i, j] = 2.0

        return state


    def get_state(self, game):
        # Take snapshot and add to stack.
        frame = self._get_single_frame(game)
        self.frames.append(frame)
        
        # Convert deque of (H, W) -> Numpy Array (4, H, W)
        return torch.from_numpy(np.array(self.frames))


    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append(state, action, reward, next_state, gameover)


    # TODO: remove this..
    #def _train_short_memory(self, state, action, reward, next_state, gameover):
    #    # Transform into tensor.
    #    state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
    #    action = torch.tensor(action, dtype=torch.float, device=self.device).unsqueeze(0)
    #    reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(0)
    #    next_state = torch.tensor(next_state, dtype=torch.float, device=self.device).unsqueeze(0)
    #    gameover = torch.tensor(gameover, dtype=torch.bool, device=self.device).unsqueeze(0)
    #    self.trainer.train_step(state, action, reward, next_state, gameover)


    def _train_long_memory(self):
        # Create batch.
        states, actions, rewards, next_states, gameovers = self.memory.sample_buffer(self.batch_size)

        # Train and return loss.
        loss = self.trainer.train_step(states, actions, rewards, next_states, gameovers)
        return loss


    def get_action(self, state):
        # The action to take.
        final_move = [0,0,0]

        # Get the 3 values Q(state, a) for each action a.
        self.model.eval()
        with torch.no_grad():
            state0 = state.detach().clone()
            state0 = torch.unsqueeze(state0, 0).to(self.device)
            prediction = self.model(state0)

        # Gready choice.
        greedy_idx = torch.argmax(prediction).item()
        # Non-gready choices.
        rnd_actions = np.delete(np.array([0, 1, 2]), greedy_idx).tolist()
        
        # Choose the move.
        if self.num_steps < 50_000:
            action = random.randint(0, 2)
            final_move[action] = 1
        elif random.random() < self.epsilon: # Exploration.
            action = np.random.choice(rnd_actions, p=[0.5, 0.5])
            final_move[action] = 1
        else: # Exploitation.
            final_move[greedy_idx] = 1

        return final_move


    def _save_checkpoint(self):
        # Store the memory to disk.
        memory_path = "./output/models/memory.h5"
        self.memory.store_buffer_h5(out_path=memory_path)

        checkpoint = {
            'episode': self.num_episodes,
            'steps': self.num_steps,
            'record': self.record,
            'epsilon': self.epsilon,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'memory_path': memory_path,
            'record_replay': self.record_replay
        }
        torch.save(checkpoint, self.out_model_path)


    def _load_checkpoint(self, checkpoint_path, load_buffer):
        # Load the checkpoint file (to CPU).
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
        # Apply saved states.
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self._target_sync()
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore metadata.
        self.num_episodes = checkpoint['episode']
        self.num_steps = checkpoint['steps']
        self.record = checkpoint['record']
        self.epsilon = checkpoint['epsilon']
        self.record_replay = checkpoint['record_replay']

        if load_buffer:
            memory_path = checkpoint['memory_path']
            self.memory.load_buffer_h5(path=memory_path)

        print(f"INFO: Checkpoint restored.")

    
    def _target_sync(self):
        """Synchronise main network and target network parameters"""
        self.target_model.load_state_dict(self.model.state_dict())


    def update_epsilon(self):
        if self.num_steps > self.memory.capacity():
            # Phase 2: Fixed epsilon
            self.epsilon = self.min_epsilon
        else:
            # Phase 1: Linear annealing
            
            # Calculate the decay factor (1.0 at start, 0.0 at end)
            decay_progress = self.num_steps / self.memory.capacity()
            
            # Calculate the current epsilon value
            self.epsilon = self.min_epsilon + (1 - self.min_epsilon) * (1 - decay_progress)


    def train(self):
        total_score = 0
        score_last_100 = deque(maxlen=100) 

        episode_steps = 0
        episode_max_loss = 0

        episode_reward = 0
        reward_last_100 = deque(maxlen=100)

        # Store info to replay the record.
        actions_replay = []
        food_replay = [self.game.food]

        # Initial state setup
        # We need to prime the stack with the first frame
        state_old = self.get_state(self.game)

        while True:

            # Store states to compute avg_q.
            if self.num_steps < 250:
                self.fixed_states[self.num_steps] = state_old

            # Choose action to perform.
            final_move = self.get_action(state_old)
 
            # Do the action, save reward, gameover and current score.
            reward, gameover, score = self.game.play_step(final_move)
            state_new = self.get_state(self.game)
 
            # Remember.
            self.remember(state_old, final_move, reward, state_new, gameover)

            # Train on batch.
            if len(self.memory) > self.batch_size:
                loss = self._train_long_memory()
                episode_max_loss = max(episode_max_loss, loss)

            # Update state_old for the next iteration.
            state_old = state_new

            # Store data for replay.
            actions_replay.append(final_move)
            food_replay.append(self.game.food)
            
            # Sync networks if necessary.
            if not self.num_steps % self.target_sync:
                print("Info: Syncronizing target model with main model...")
                self._target_sync()

            # Decaying epsilon.
            self.update_epsilon()

            # Update some stats.
            self.num_steps += 1
            episode_steps += 1
            episode_reward += reward

    
            if gameover:
                # Reset game.
                self.game.reset()
                self.num_episodes += 1

                # Reset stack and take snapshot.
                self._reset_stack()
                state_old = self.get_state(self.game)

                # Save checkpoint if new record reached.
                if score > self.record:
                    self.record = score
                    self.record_replay['actions'] = actions_replay
                    self.record_replay['foods'] = food_replay

                    # Create checkpoint.
                    print("INFO: New record! Saving checkpoint...")
                    self._save_checkpoint()


                # Reset replay.
                actions_replay = []
                food_replay = []
        
                # Update stats.
                total_score += score
                mean_score = total_score / self.num_episodes
                score_last_100.append(score)
                mean_score_100 = sum(score_last_100) / len(score_last_100)

                reward_last_100.append(episode_reward)
                mean_reward_100 = sum(reward_last_100) / len(reward_last_100)


                self.model.eval()
                with torch.no_grad():
                    avg_q = self.model(self.fixed_states)
                    avg_q = torch.max(avg_q, dim=1).values # shape: N
                    avg_q = avg_q.mean().item()

                print(
                    "INFO:\n"
                    f"\tGAME: {self.num_episodes}\n"
                    f"\tRecord: {self.record}\n"
                    f"\tSteps: {self.num_steps}\n"
                    f"\tBuffer memory size: {len(self.memory)}\n"
                    f"\tScore: {score}\n"
                    f"\tDuration (steps): {episode_steps}\n"
                    #f"\tMean score: {mean_score}\n"
                    f"\tMean score last 100: {mean_score_100}\n"
                    f"\tMean reward last 100: {mean_reward_100}\n"
                    f"\tTotal score: {total_score}\n"
                    f"\tepsilon: {self.epsilon}"
                )

                csv_line = f"{mean_score},{mean_score_100},{score},{mean_reward_100},{self.epsilon},{episode_max_loss},{avg_q}\n"

                # Save stats in csv.
                if self.out_csv_path is not None:
                    with open(self.out_csv_path, 'a') as f:
                        f.write(csv_line)

                # Reset stats.
                episode_steps = 0
                episode_max_loss = 0
                episode_reward = 0
