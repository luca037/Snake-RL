from Agent import *
from Game import ReplaySnakeGame, Point

MAX_DATASET_SIZE = int(1_000_000 / 4)
BATCH_SIZE = 32
LR = 0.00025
RANDOM_STEPS = 50_000

EPSILON = 1
DECAYING_FACTOR = 0.995
MIN_EPSILON = 0.01 / 10
GAMMA = 0.99

TARGET_SYNC = 5_000

OUT_MODEL_FILE_PATH = "./output/models/model.pth"
OUT_CSV_PATH = "./output/csv/stats.csv"


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
    #agent = BlindAgent(
    #    max_dataset_size = MAX_DATASET_SIZE,
    #    batch_size       = BATCH_SIZE,
    #    lr               = LR,
    #    epsilon          = EPSILON,
    #    decaying_epsilon = DECAYING_FACTOR,
    #    min_epsilon      = MIN_EPSILON,
    #    gamma            = GAMMA,
    #    target_sync      = TARGET_SYNC,
    #    out_model_path   = OUT_MODEL_FILE_PATH,
    #    out_csv_path     = OUT_CSV_PATH,
    #    device           = device,
    #    gui              = False,
    #    checkpoint_path  = None
    #)
    #agent.train()


    # Lidar AGENT.
    #agent = LidarAgent(
    #    max_dataset_size = MAX_DATASET_SIZE,
    #    batch_size       = BATCH_SIZE,
    #    lr               = LR,
    #    epsilon          = EPSILON,
    #    min_epsilon      = MIN_EPSILON,
    #    gamma            = GAMMA,
    #    target_sync      = TARGET_SYNC,
    #    out_model_path   = OUT_MODEL_FILE_PATH,
    #    out_csv_path     = OUT_CSV_PATH,
    #    device           = device,
    #    gui              = False,
    #    checkpoint_path  = OUT_MODEL_FILE_PATH,
    #    load_buffer      = True 
    #)
    ##print(len(agent.memory))
    #agent.train()

    # ATARI AGENT.
    #agent = AtariAgent(
    #    max_dataset_size = MAX_DATASET_SIZE,
    #    batch_size       = BATCH_SIZE,
    #    random_steps     = RANDOM_STEPS,
    #    lr               = LR,
    #    epsilon          = EPSILON,
    #    min_epsilon      = MIN_EPSILON,
    #    gamma            = GAMMA,
    #    target_sync      = TARGET_SYNC,
    #    out_model_path   = OUT_MODEL_FILE_PATH,
    #    out_csv_path     = OUT_CSV_PATH,
    #    device           = device,
    #    gui              = False,
    #    checkpoint_path  = OUT_MODEL_FILE_PATH,
    #    load_buffer      = False 
    #)
    ##agent.num_steps = 300_00000
    ##print(len(agent.memory))
    #agent.train()


    agent = Cerberus(
        model1="./output/models/model_atari_61.pth",
        model2="./output/models/model_start19_plus45.pth",
        model3="./output/models/model_start47_plus39.pth",
        device=device,
        gui=False,
        out_model_path=OUT_MODEL_FILE_PATH,
        checkpoint_path=OUT_MODEL_FILE_PATH
    )
    #agent.train()

    print(agent.record)
    actions = agent.record_replay['actions']
    foods = agent.record_replay['foods']
    replay = ReplaySnakeGame(foods)
    for action in actions:
        replay.play_step(action)
