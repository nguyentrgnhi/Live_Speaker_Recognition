import multiprocessing
import datetime
import os
import torch

TRAIN_DATA_DIR = os.path.join(os.path.expanduser('~'), r"/Users/nguyennhi/PycharmProjects/pythonProject1/train-clean-100")
TEST_DATA_DIR = os.path.join(os.path.expanduser('~'), r"/Users/nguyennhi/PycharmProjects/pythonProject1/test-clean")
MY_TEST_DATA_DIR = os.path.join(os.path.expanduser('~'), r"/Users/nguyennhi/PycharmProjects/pythonProject1/test")


timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
filename = f"saved_model_{timestamp}.pt"
SAVED_MODEL_PATH = os.path.join(os.path.expanduser('~'), r"/Users/nguyennhi/PycharmProjects/pythonProject1",filename)

BATCH_SIZE = 10
NUM_PROCESSES = min(multiprocessing.cpu_count(), BATCH_SIZE)

TRAINING_STEPS = 100

USE_TRANSFORMER = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_MFCC = 40
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 3
BI_LSTM = True
FRAME_AGGREGATION_MEAN = True
LEARNING_RATE = 0.0001
SEQ_LEN = 100
SPECAUG_TRAINING = False
SAVE_MODEL_FREQUENCY=10

SPECAUG_FREQ_MASK_PROB = 0.3
SPECAUG_TIME_MASK_PROB = 0.3
SPECAUG_FREQ_MASK_MAX_WIDTH = N_MFCC // 5
SPECAUG_TIME_MASK_MAX_WIDTH = SEQ_LEN // 5
TRIPLET_ALPHA = 0.1

USE_TRANSFORMER = False
TRANSFORMER_DIM = 32
TRANSFORMER_ENCODER_LAYERS = 2
TRANSFORMER_HEADS = 8
NUM_EVAL_TRIPLETS = 100


USE_FULL_SEQUENCE_INFERENCE = False
SLIDING_WINDOW_STEP = 50
EVAL_THRESHOLD_STEP = 0.001
