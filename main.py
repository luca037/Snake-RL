from Agent import *
from Game import ReplaySnakeGame, Point

MAX_DATASET_SIZE = 1_000_000
BATCH_SIZE = 32
LR = 0.00025

EPSILON = 1
DECAYING_FACTOR = 0.995
MIN_EPSILON = 0.01
GAMMA = 0.99

TARGET_SYNC = 10_000

OUT_MODEL_FILE_PATH = "./output/models/model.pth"
OUT_CSV_PATH = "./output/csv/avg_score2.csv"

if __name__ == "__main__": 

    # Define the device variable.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("INFO: CUDA is available. Running on GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print ("INFO: MPS device found. Running on GPU")
    else:
        device = torch.device("cpu")
        print("INFO: CUDA and MPS not available. Running on CPU.")

    # BLIND AGENT.
    #agent = FFNNAgent(
    #    max_dataset_size = MAX_DATASET_SIZE,
    #    batch_size       = BATCH_SIZE,
    #    lr               = LR,
    #    epsilon          = EPSILON,
    #    decaying_epsilon = DECAYING_FACTOR,
    #    min_epsilon      = MIN_EPSILON,
    #    gamma            = GAMMA,
    #    out_model_path   = OUT_MODEL_FILE_PATH,
    #    out_csv_path     = OUT_CSV_PATH,
    #    device           = device,
    #    gui              = False,
    #    checkpoint_path  = OUT_MODEL_FILE_PATH
    #)
    #agent.train()

    # ATARI AGENT.
    agent = AtariAgent(
        max_dataset_size = MAX_DATASET_SIZE,
        batch_size       = BATCH_SIZE,
        lr               = LR,
        epsilon          = EPSILON,
        decaying_epsilon = DECAYING_FACTOR,
        min_epsilon      = MIN_EPSILON,
        gamma            = GAMMA,
        target_sync      = TARGET_SYNC,
        out_model_path   = OUT_MODEL_FILE_PATH,
        out_csv_path     = OUT_CSV_PATH,
        device           = device,
        gui              = False,
        checkpoint_path  = OUT_MODEL_FILE_PATH
    )
    agent.train()

    actions = agent.record_replay['actions']
    foods = agent.record_replay['foods']
    replay = ReplaySnakeGame(foods)
    for action in actions:
        replay.play_step(action)


# TODOS:
# - definire statistiche utili per studiare il comportamento al variare
#   dei parametri. (e.g. mean score, score received, etc.)
# - piccoli test facendo variare alcuni dei parametri (e.g. epsilon start)
# - funzione 'replay' per ricreare il record del modello.
