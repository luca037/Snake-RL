from collections import deque
import random
import numpy as np
from Game import SnakeGame, Direction, Point, BLOCK_SIZE
from Model import Linear_QNet, QTrainer

MAX_DATASET_SIZE = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    
    def __init__(self):
        # Exploration / Exploitation trade-off.
        self.epsilon = 0.1

        # Disount factor.
        self.gamma = 0.9

        # Episodes counter.
        self.num_episodes = 0

        # Dataset with pairs (state, target)
        self.memory = deque(maxlen=MAX_DATASET_SIZE)

        # Action-value function approximator.
        self.model = Linear_QNet(11, 256, 3)

        # Model trainer.
        self.trainer = QTrainer(self.model, LR, self.gamma)
    

    def get_state(self, game):
        # Get snake head.
        head = game.snake[0]

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


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def train_long_memory(self):
        # Define the batch.
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE)
        else:
            batch = self.memory

        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        #self.epsilon = 80 - self.num_episodes
        final_move = [0,0,0]

        #if random.randint(0, 200) < self.epsilon:
        #    move = random.randint(0, 2)
        #    final_move[move] = 1
        #else:
        #    state0 = np.array(state, dtype=float)
        #    state0 = np.expand_dims(state0, 0)
        #    prediction = self.model(state0, training=False)
        #    move = np.argmax(prediction[0]).item()
        #    final_move[move] = 1

        greedy = np.random.choice([1, 0], p=[1-self.epsilon, self.epsilon])

        state0 = np.array(state, dtype=float)
        state0 = np.expand_dims(state0, 0)
        prediction = self.model(state0, training=False)

        greedy_idx = np.argmax(prediction[0]).item()
        rnd_idx = np.delete(np.array([0, 1, 2]), greedy_idx).tolist()
        
        if greedy:
            final_move[greedy_idx] = 1
        else:
            idx = np.random.choice(rnd_idx, p=[0.5, 0.5])
            final_move[idx] = 1


        #print("pred", prediction)
        #print("greedy idx:", greedy_idx)
        #print("rnd idx:", rnd_idx)
        #print("Is greedy?", greedy)
        #print("final move:", final_move)

        return final_move
  

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.num_episodes += 1
            agent.train_long_memory()

            if score > record:
                record = score
                #agent.model.save()

            # Decaying epsilon.
            if (agent.num_episodes % 50 == 0):
                print("eps =",agent.epsilon)
                agent.epsilon = max(0.01, agent.epsilon - 0.02)
                

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_episodes
            plot_mean_scores.append(mean_score)
            #plot(plot_scores, plot_mean_scores)

            print('Game', agent.num_episodes, 'Score', score, 'Record:', record, "Mean score:", mean_score)


if __name__ == "__main__":
    #agent = Agent()
    #agent.model.summary()

    #x = np.random.rand(2, 11)
    #print(x.shape)
    #y = agent.model.predict(x)
    #print(y.shape)
    #print(y)

    train()

