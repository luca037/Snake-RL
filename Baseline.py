import os
from Game import SnakeGame, Point, BLOCK_SIZE

class Baseline:
    def __init__(self, gui=False, out_csv_path = None):

        # Init the game.
        self.game = SnakeGame(gui=gui)
        self.rows = self.game.h // BLOCK_SIZE
        self.cols = self.game.w // BLOCK_SIZE

        # Steps counter (total number of actions taken).
        self.steps = 0

        # Where stats are stored.
        self.out_csv_path = out_csv_path

        # Create csv file with stats, if necessary.
        if self.out_csv_path is not None:
            csv_header = "score,steps\n"
            if not os.path.exists(self.out_csv_path):
                with open(self.out_csv_path, 'w') as f:
                    f.write(csv_header)
            print("INFO: Stats will be stored in", self.out_csv_path)


    def play_action(self, action):
        old_score = self.game.score
        _, gameover, score = self.game.play_step(action)
        self.steps += 1

        if score - old_score and self.out_csv_path is not None:
            with open(self.out_csv_path, 'a') as f:
                f.write(f"{score},{self.steps}\n")

        return gameover


    def visit_all_cols(self):
        i = self.game.head.y // BLOCK_SIZE
        j = self.game.head.x // BLOCK_SIZE

        while True:

            if j % 2: # If odd cols.
                while i < self.rows - 2: 
                    action = [1, 0, 0] # Go straight.
                    if self.play_action(action): break
                    i += 1
                #i = 0 # Reset row counter.
                if j != self.cols - 1:
                    action = [0, 0, 1] # Turn left.
                    if self.play_action(action): break
                    if self.play_action(action): break
                    i -= 1 # Caused by second turn left action.
                    j += 1 # Go to next col.
                else:
                    action = [1, 0, 0] # Go straight.
                    if self.play_action(action): break
                    action = [0, 1, 0] # Turn right.
                    if self.play_action(action): break
                    while j > 1:
                        action = [1, 0, 0] # Go straight.
                        if self.play_action(action): break
                        j -= 1
                    action = [0, 1, 0] # Turn right.
                    if self.play_action(action): break
                    i = self.rows - 2 # Reset rows conuter.
                    j = 0 # Reset cols counter.

            else:
                while i > 0:
                    action = [1, 0, 0] # Go straight.
                    if self.play_action(action): break
                    i -= 1
                action = [0, 1, 0] # Turn right.
                if self.play_action(action): break
                if self.play_action(action): break
                i += 1 # Caused by second turn right action.
                j += 1 # Go to next col.


    def play(self):
        i = self.game.head.y // BLOCK_SIZE
        j = self.game.head.x // BLOCK_SIZE

        # Assuming starting direction == RIGHT.
        if j % 2:
            action = [0, 1, 0] # Turn right.
        else:
            action = [0, 0, 1] # Turn left.
        self.game.play_step(action)

        self.visit_all_cols()

        self.game.reset()
        self.steps = 0
