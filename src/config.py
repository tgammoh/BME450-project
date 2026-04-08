# Data parameters
WINDOW_SIZE       = 200       # samples per window (100ms at 2000Hz)
STEP_SIZE         = 50        # step between windows (25ms)
SAMPLING_RATE     = 2000      # Hz

# Split 
TRAIN_REPS        = [1, 3, 4, 6]
TEST_REPS         = [2, 5]
PURITY_THRESHOLD  = 0.9       # min fraction of window that must agree on label

# Training parameters 
BATCH_SIZE        = 32
LEARNING_RATE = 0.0003
NUM_EPOCHS = 60
DROPOUT = 0.1

# Dataset parameters
NUM_CLASSES       = 18        # 0=rest, 1-17=movements
NUM_CHANNELS      = 12        # EMG electrodes


