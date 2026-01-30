from Agent import *
from Game import ReplaySnakeGame, Point, SnakeGame
from Baseline import Baseline

import argparse


######################### Hyperparameters #####################################
MAX_DATASET_SIZE = int(1_000_000 / 4)
BATCH_SIZE = 32
LR = 0.00025
RANDOM_STEPS = 50_000

EPSILON = 1
DECAYING_FACTOR = 0.995
MIN_EPSILON = 0.01 / 100
GAMMA = 0.99

TARGET_SYNC = 5_000

########################### Output files ######################################

OUT_MODEL_FILE_PATH = "./output/models/model_LidarAgent.pth"
OUT_CSV_PATH = "./output/csv/stats.csv"

######################### Pre-trained models ##################################

# Paths of the checkpoints.
ATARI_PATH = "./output/models/model_atari_61.pth"
BLIND_PATH = "./output/models/model_BlindAgent.pth"
LIDAR_PATH = "./output/models/model_LidarAgent.pth"

CERBERUS_H1 = "./output/models/model_atari_61.pth"
CERBERUS_H2 = "./output/models/model_start19_plus45.pth"
CERBERUS_H3 = "./output/models/model_start47_plus44.pth"
CERBERUS_RECORD = "./output/models/model_RECORD_97.pth"

# Paths of the replay buffers (.h5 files).:w
MEMORY_ATARI = "./output/models/memory_AtariAgent.h5"
MEMORY_BLIND = "./output/models/memory_BlindAgent.h5"
MEMORY_LIDAR = "./output/models/memory_LidarAgent.h5"

##############################################################################


def main():

    # Define command line args.
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="none", help="Specify agent type. Values: atari, blind, lidar or baseline.")
    parser.add_argument("--train", action="store_true", help="If you want to train the agent.")
    parser.add_argument("--loadbuf", action="store_true", help="If you want to load the buffer to 'device'.")
    parser.add_argument("--record", action="store_true", help="If you want to watch the record of the agent.")
    parser.add_argument("--nogui", action="store_false", help="If you want to deactivate the gui.")

    # Parse args.
    args = parser.parse_args()

    BuildAgent = None
    if args.agent == "blind":
        BuildAgent = BlindAgent
        checkpoint = BLIND_PATH
        memory = MEMORY_BLIND
    elif args.agent == "lidar":
        BuildAgent = LidarAgent
        checkpoint = LIDAR_PATH
        memory = MEMORY_LIDAR
    elif args.agent == "atari":
        BuildAgent = AtariAgent 
        checkpoint = ATARI_PATH
        memory = MEMORY_ATARI

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


    if BuildAgent is not None:
        agent = BuildAgent(
            max_dataset_size = MAX_DATASET_SIZE,
            batch_size       = BATCH_SIZE,
            lr               = LR,
            epsilon          = EPSILON,
            decaying_epsilon = DECAYING_FACTOR,
            min_epsilon      = MIN_EPSILON,
            gamma            = GAMMA,
            target_sync      = TARGET_SYNC,
            out_model_path   = OUT_MODEL_FILE_PATH,
            memory_path      = memory,
            out_csv_path     = OUT_CSV_PATH,
            device           = device,
            gui              = args.nogui,
            checkpoint_path  = checkpoint,
            load_buffer      = args.loadbuf
        )
        print("INFO: Selected agent:", args.agent)
    elif args.agent == "cerberus":
        agent = CerberusAgent(
            head1=CERBERUS_H1,
            head2=CERBERUS_H2,
            head3=CERBERUS_H3,
            device=device,
            gui=args.nogui,
            out_model_path=OUT_MODEL_FILE_PATH,
            checkpoint_path=CERBERUS_RECORD,
            out_csv_path=OUT_CSV_PATH,
            out_csv_path2="./output/csv/cerberus_efficiency.csv",
        )

    if args.train:
        agent.train()
    elif args.record:
        print("INFO: Record score of the agent:", agent.record)
        print("INFO: Showing record...")
        actions = agent.record_replay['actions']
        foods = agent.record_replay['foods']
        replay = ReplaySnakeGame(foods)
        for action in actions:
            replay.play_step(action)
    elif args.agent == "baseline":
        print("INFO: Baseline algorithm will play...")
        baseline = Baseline(
            gui=args.gui,
            out_csv_path="./output/csv/baseline.csv"
        )
        baseline.play()


if __name__ == "__main__": 
    main()
