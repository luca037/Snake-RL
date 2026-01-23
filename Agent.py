from collections import deque
import random
import numpy as np
import torch
from Game import SnakeGame, Direction, Point, BLOCK_SIZE
from Model import *
import os
from Utils import ReplayBuffer
from abc import ABC, abstractmethod

class BaseAgent(ABC):

    def __init__(self,
            max_dataset_size = 10,
            batch_size       = 32,
            random_steps     = 0,
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
            checkpoint_path  = None,
            load_buffer      = False
    ):
        # Hyperparams.
        self.gamma = gamma
        self.epsilon = epsilon
        self.decaying_epsilon = decaying_epsilon
        self.min_epsilon = min_epsilon
        self.random_steps = random_steps
        self.lr = lr
        self.target_sync = target_sync
        self.max_dataset_size = max_dataset_size
        self.batch_size = batch_size
        self.device = device

        # Paths.
        self.out_model_path = out_model_path
        self.out_csv_path = out_csv_path

        # Game and stats.
        self.game = SnakeGame(gui=gui)
        self.record = 0
        self.record_replay = {'actions': [], 'foods': []}
        self.num_episodes = 0
        self.num_steps = 0

        # Init all necessary components.
        self.memory = self._init_memory()
        self.model, self.target_model, self.trainer = self._init_model()

        # Used to comptue action-value function.
        self.fixed_states = None

        # Load checkpoint_path, if necesary.
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, load_buffer)

        # Create csv file with stats, if necessary.
        if self.out_csv_path is not None:
            csv_header = "mean_score,mean_score_100,score,mean_reward_100,epsilon,max_loss,avg_q\n"
            if not os.path.exists(self.out_csv_path):
                with open(self.out_csv_path, 'w') as f:
                    f.write(csv_header)
            print("INFO: Stats will be stored in", self.out_csv_path)


    ### Abstract methods ###
    @abstractmethod
    def _init_memory(self):
        pass

    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def _update_epsilon(self):
        pass

    @abstractmethod
    def get_state(self, game):
        pass

    def _on_reset(self):
        pass

    ### Common methods ###
    def _save_checkpoint(self):
        # Store the memory to disk.
        memory_path = "./output/models/memory.h5" # TODO: don't hardcode.
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


    def _remember(self, state, action, reward, next_state, gameover):
        self.memory.append(state, action, reward, next_state, gameover)


    def _target_sync(self):
        self.target_model.load_state_dict(self.model.state_dict())


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
        if self.num_steps < self.random_steps:
            action = random.randint(0, 2)
            final_move[action] = 1
        elif random.random() < self.epsilon: # Exploration.
            action = np.random.choice(rnd_actions, p=[0.5, 0.5])
            final_move[action] = 1
        else: # Exploitation.
            final_move[greedy_idx] = 1

        return final_move


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
        state_old = self.get_state(self.game)

        # To monitor action value (Q)
        if self.fixed_states is None:
            self.fixed_states = torch.zeros((250, *state_old.shape), device=self.device)

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
            self._remember(state_old, final_move, reward, state_new, gameover)

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
            self._update_epsilon()

            # Update some stats.
            self.num_steps += 1
            episode_steps += 1
            episode_reward += reward

    
            if gameover:
                # Reset game.
                self.game.reset()
                self.num_episodes += 1

                # Reset stack and take snapshot.
                self._on_reset()
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


class BlindAgent(BaseAgent):
    
    def _init_memory(self):
        return ReplayBuffer(
            state_shape=(11, ),
            action_dim=3,
            max_size=self.max_dataset_size,
            device=self.device
        )
    

    def _init_model(self):
        model = Linear_QNet(11, 128, 3).to(self.device)
        target_model = Linear_QNet(11, 128, 3).to(self.device)
        trainer = QTrainer(model, target_model, self.lr, self.gamma)
        return model, target_model, trainer


    def _update_epsilon(self):
        # Decaying epsilon.
        if not self.num_steps % 10:
            self.epsilon = max(self.epsilon * self.decaying_epsilon, self.min_epsilon)


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
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y  # food down
        ]

        return torch.tensor(state, dtype=torch.float)


class LidarAgent(BaseAgent):
    
    def _init_memory(self):
        return ReplayBuffer(
            state_shape=(28, ),
            action_dim=3,
            max_size=self.max_dataset_size,
            device=self.device
        )


    def _init_model(self):
        model = Linear_QNet(28, 128, 3).to(self.device)
        target_model = Linear_QNet(28, 128, 3).to(self.device)
        trainer = QTrainer(model, target_model, self.lr, self.gamma)
        return model, target_model, trainer


    def get_state(self, game):
        head = game.snake[0]
        
        # Define the 8 directions.
        directions = [
            (-BLOCK_SIZE, 0),           # Left
            (-BLOCK_SIZE, -BLOCK_SIZE), # Up-Left
            (0, -BLOCK_SIZE),           # Up
            (BLOCK_SIZE, -BLOCK_SIZE),  # Up-Right
            (BLOCK_SIZE, 0),            # Right
            (BLOCK_SIZE, BLOCK_SIZE),   # Down-Right
            (0, BLOCK_SIZE),            # Down
            (-BLOCK_SIZE, BLOCK_SIZE)   # Down-Left
        ]
    
        # Current state.
        state = []
    
        # For each direction.
        for dx, dy in directions:
            
            # Init starting position (aka snake head).
            x, y = head.x, head.y
            
            distance = 0    # From wall.
            food_found = 0  # 1=>True, 0=>False.
            body_found = 0  # 1=>True, 0=>False.
            
            # Check each position along current direction.
            while True:
                # Move to next position along current direction.
                x += dx
                y += dy
                distance += 1
                
                # Collision with wall check.
                if x < 0 or x >= game.w or y < 0 or y >= game.h:
                    wall_inv_dist = 1.0 / distance
                    break  # Stop looking in this direction, go to next one.
                
                # Food check.
                if food_found == 0 and x == game.food.x and y == game.food.y:
                    food_found = 1
                
                # Body check.
                if body_found == 0:
                    for point in game.snake[1:]:
                        if x == point.x and y == point.y:
                            body_found = 1
                            break # Stop when you see nearest piece.
    
            # Store the triplets associated to current direction.
            state.append(food_found)
            state.append(body_found)
            state.append(wall_inv_dist)


        # Append current direction to state.
        state.append(game.direction == Direction.LEFT)
        state.append(game.direction == Direction.RIGHT)
        state.append(game.direction == Direction.UP)
        state.append(game.direction == Direction.DOWN)
    
        return torch.tensor(state, dtype=torch.float)


    def _update_epsilon(self):
        if self.num_steps > self.memory.capacity():
            # Phase 2: Fixed epsilon
            self.epsilon = self.min_epsilon
        else:
            # Phase 1: Linear annealing
            
            # Calculate the decay factor (1.0 at start, 0.0 at end)
            decay_progress = self.num_steps / self.memory.capacity() 
            
            # Calculate the current epsilon value
            self.epsilon = self.min_epsilon + (1 - self.min_epsilon) * (1 - decay_progress)


class AtariAgent(BaseAgent):

    def __init__(self, **kwargs):
        # Define the frame stack.
        self.frame_rows = 12
        self.frame_cols = 12
        self.stack_size = 4
        self.frames = deque(maxlen=self.stack_size)
        self._on_reset()
        super().__init__(**kwargs)


    def _init_memory(self):
        return ReplayBuffer(
            state_shape=(self.stack_size, self.frame_rows, self.frame_cols),
            action_dim=3,
            max_size=self.max_dataset_size,
            device=self.device
        )
 

    def _init_model(self):
        model = CNN_QNet(
                in_chan=4,
                grid_rows=self.frame_rows,
                grid_cols=self.frame_cols
        ).to(self.device)

        target_model = CNN_QNet(
                in_chan=4,
                grid_rows=self.frame_rows,
                grid_cols=self.frame_cols
        ).to(self.device)

        trainer = QTrainer(model, target_model, self.lr, self.gamma)

        return model, target_model, trainer


    def _update_epsilon(self):
        if self.num_steps > self.memory.capacity():
            # Phase 2: Fixed epsilon
            self.epsilon = self.min_epsilon
        else:
            # Phase 1: Linear annealing
            
            # Calculate the decay factor (1.0 at start, 0.0 at end)
            decay_progress = self.num_steps / self.memory.capacity()
            
            # Calculate the current epsilon value
            self.epsilon = self.min_epsilon + (1 - self.min_epsilon) * (1 - decay_progress)


    def _on_reset(self):
        # Reset stack.
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
        
        # Convert deque to torch.tensor.
        return torch.from_numpy(np.array(self.frames))


class CerberusAgent(BaseAgent):
    def __init__(self,
            device          = 'cpu',
            model1          = None,
            model2          = None,
            model3          = None,
            gui             = False,
            out_model_path  = "./model.pth",
            checkpoint_path = None
    ):
        self.device = device

        # The 3 heads.
        self.agent1 = AtariAgent(device=self.device, checkpoint_path=model1, epsilon=-1, load_buffer=False, max_dataset_size=0)
        self.agent2 = AtariAgent(device=self.device, checkpoint_path=model2, epsilon=-1, load_buffer=False, max_dataset_size=0)
        self.agent3 = AtariAgent(device=self.device, checkpoint_path=model3, epsilon=-1, load_buffer=False, max_dataset_size=0)


        # Need to overwrite the checkpoing thing.
        self.agent1.epsilon = -1
        self.agent2.epsilon = -1
        self.agent3.epsilon = -1

        # The game.
        self.game = SnakeGame(gui=gui)

        # Metadata.
        self.num_steps = 0
        self.num_episodes = 0
        self.record = 0
        self.record_replay = {'actions': [], 'foods': []}


        # Where checkpoint is stored.
        self.out_model_path = out_model_path

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)


    # CerberusAgent agent is not proper agent: we don't train it.
    # So we need to raise error for the non implemented functions.
    def _init_model(self):
        raise NotImplementedError

    def _init_memory(self):
        raise NotImplementedError

    def _update_epsilon(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def _remember(self, state, action, reward, next_state, gameover):
        raise NotImplementedError

    def _target_sync(self):
        raise NotImplementedError

    def _train_long_memory(self):
        raise NotImplementedError


    def _save_checkpoint(self):
        checkpoint = {
            'episode': self.num_episodes,
            'steps': self.num_steps,
            'record': self.record,
            'record_replay': self.record_replay
        }
        torch.save(checkpoint, self.out_model_path)


    def _load_checkpoint(self, checkpoint_path):
        # Load the checkpoint file (to CPU).
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
        
        # Restore metadata.
        self.num_episodes = checkpoint['episode']
        self.num_steps = checkpoint['steps']
        self.record = checkpoint['record']
        self.record_replay = checkpoint['record_replay']
        print(f"INFO: Checkpoint restored.")


    def get_action(self, state, model):
        # The action to take.
        final_move = [0,0,0]

        # Get the 3 values Q(state, a) for each action a.
        model.eval()
        with torch.no_grad():
            state0 = state.detach().clone()
            state0 = torch.unsqueeze(state0, 0).to(self.device)
            prediction = model(state0)

        # Gready choice.
        greedy_idx = torch.argmax(prediction).item()

        # Non-gready choices.
        rnd_actions = np.delete(np.array([0, 1, 2]), greedy_idx).tolist()
        
        final_move[greedy_idx] = 1
        
        if self.game.is_collision(self.game.move(final_move, perform=False)):
            #print("Greedy is collision => change action")
            a = max(prediction[0, rnd_actions[0]].item(), prediction[0, rnd_actions[1]].item())
            b = min(prediction[0, rnd_actions[0]].item(), prediction[0, rnd_actions[1]].item())

            opt_a = [0, 0, 0]
            opt_b = [0, 0, 0]
            opt_a[rnd_actions[0]] = 1
            opt_b[rnd_actions[1]] = 1

            if not self.game.is_collision(self.game.move(opt_a, perform=False)):
                final_move = opt_a
                #print("changed with a")
            elif not self.game.is_collision(self.game.move(opt_b, perform=False)):
                final_move = opt_b
                #print("changed with b")

        return final_move


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
        state_old = self.agent1.get_state(self.game)

        score = 0

        while True:

            # Choose action to perform.
            if score < 25:
                final_move = self.get_action(state_old, self.agent1.model)
            elif score < 38:
                final_move = self.get_action(state_old, self.agent2.model)
            else:
                final_move = self.get_action(state_old, self.agent3.model)
 
            # Do the action, save reward, gameover and current score.
            reward, gameover, score = self.game.play_step(final_move)
            state_new = self.agent1.get_state(self.game)

            # Update state_old for the next iteration.
            state_old = state_new

            # Store data for replay.
            actions_replay.append(final_move)
            food_replay.append(self.game.food)

            # Update some stats.
            self.num_steps += 1
            episode_steps += 1
            episode_reward += reward

            if gameover:
                # Reset game.
                self.game.reset()
                self.num_episodes += 1

                # Reset stack and take snapshot.
                self.agent1._on_reset()
                state_old = self.agent1.get_state(self.game)

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

                print(
                    "INFO:\n"
                    f"\tGAME: {self.num_episodes}\n"
                    f"\tRecord: {self.record}\n"
                    f"\tSteps: {self.num_steps}\n"
                    #f"\tBuffer memory size: {len(self.memory)}\n"
                    f"\tScore: {score}\n"
                    f"\tDuration (steps): {episode_steps}\n"
                    #f"\tMean score: {mean_score}\n"
                    f"\tMean score last 100: {mean_score_100}\n"
                    f"\tMean reward last 100: {mean_reward_100}\n"
                    f"\tTotal score: {total_score}\n"
                    f"\tAgent1 memory size: {len(self.agent1.memory)}\n"
                    f"\tAgent2 memory size: {len(self.agent2.memory)}\n"
                    f"\tAgent3 memory size: {len(self.agent3.memory)}\n"
                    #f"\tepsilon: {self.epsilon}"
                )

                #csv_line = f"{mean_score},{mean_score_100},{score},{mean_reward_100},{self.epsilon},{episode_max_loss},{avg_q}\n"

                ## Save stats in csv.
                #if self.out_csv_path is not None:
                #    with open(self.out_csv_path, 'a') as f:
                #        f.write(csv_line)

                # Reset stats.
                episode_steps = 0
                episode_max_loss = 0
                episode_reward = 0



