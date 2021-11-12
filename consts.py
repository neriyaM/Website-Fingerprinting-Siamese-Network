########### FILES & DIRS ###########
CONFIG_FILE_NAME = "config.json"
TENSORBOARD_LOGS_DIR = "TensorBoardLogs"
RESULTS_FILE = "results.csv"
SAMPLES_DIR = "samples"
SAMPLES_CORRECT_DIR = "correct"
SAMPLES_INCORRECT_DIR = "incorrect"

########### SIZES & DIMS ###########
INPUT_SHAPE_GRAYSCALE = (148, 1)

#PATHS
PAIRS_TRAIN_PATH = ""
PAIRS_TEST_PATH = ""

########### KEYS ###########
RUNNERS_KEY = "Runners"
RUNNER_NAME_KEY = "name"
BATCH_SIZE_KEY = "batch_size"
VALIDATION_SIZE_KEY = "validation_size"
EPOCHS_KEY = "epochs"
BATCHNORM_KEY = "batchnorm"
DROPOUT_KEY = "dropout"
OPTIMIZER_KEY = "optimizer"
LEARNING_RATE_KEY = "learning_rate"
CALLBACKS_KEY = "callbacks"
TEST_ACCURACY_KEY = "test_accuracy"
DURATION_KEY = "duration"

########### RESULTS ###########
RESULTS_FIELDS = [BATCH_SIZE_KEY, VALIDATION_SIZE_KEY, EPOCHS_KEY, BATCHNORM_KEY, DROPOUT_KEY,
                  OPTIMIZER_KEY, LEARNING_RATE_KEY, TEST_ACCURACY_KEY, DURATION_KEY]
