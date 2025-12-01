from Agent import *

MAX_DATASET_SIZE = 100_000
BATCH_SIZE = 500
LR = 0.001

EPSILON = 1
DECAYING_FACTOR = 0.95
MIN_EPSILON = 0.02
GAMMA = 0.9

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
        print("INFO: CUDA and MMPS not available. Running on CPU.")

    agent = FFNNAgent(
        max_dataset_size = MAX_DATASET_SIZE,
        batch_size = BATCH_SIZE,
        lr = LR,
        epsilon = EPSILON,
        decaying_epsilon = DECAYING_FACTOR,
        min_epsilon = MIN_EPSILON,
        gamma = GAMMA,
        out_model_path = OUT_MODEL_FILE_PATH,
        out_csv_path = OUT_CSV_PATH,
        device = device,
        gui = True,
        checkpoint_path = None
    )
    agent.train()


# TODOS:
# - definire statistiche utili per studiare il comportamento al variare
#   dei parametri. (e.g. mean score, score received, etc.)
# - piccoli test facendo variare alcuni dei parametri (e.g. epsilon start)
# - funzione 'replay' per ricreare il record del modello.
