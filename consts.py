########### FILES & DIRS ###########
CONFIG_FILE_NAME = "config.json"
TENSORBOARD_LOGS_DIR = "TensorBoardLogs"
RESULTS_FILE = "results.csv"

########### SIZES & DIMS ###########
INPUT_SIZES = {
    "data": (1000, 1),
    "data_old": (148, 1),
    "var_cnn": (1000, 1)
}

########### KEYS ###########
RUNNERS_KEY = "Runners"
BASE_MODELS_KEY = "base_models"
DATASETS_KEY = "datasets"
EMBEDDING_SIZES_KEY = "embedding_sizes"
BATCH_SIZES_KEY = "batch_sizes"
DISTANCE_LAMBDAS_KEY = "distance_lambdas"
OPTIMIZERS_KEY = "optimizers"
NAME_KEY = "name"
LEARNING_RATES_KEY = "learning_rates"
VALIDATION_SIZES_KEY = "validation_sizes"
EPOCHS_KEY = "epochs"
CALLBACKS_KEY = "callbacks"
DROPOUT_KEY = "dropout"
BATCHNORM_KEY = "batchnorm"

TRAIN_KEY = "train"
TEST_KEY = "test"
PAIRS_KEY = "pairs"
FEATURES_KEY = "features"
MATCH_FILE_KEY = "match.csv"
MISMATCH_FILE_KEY = "mismatch.csv"

TEST_ACCURACY_KEY = "test_accuracy"
DURATION_KEY = "duration"

########### RESULTS ###########
RESULTS_FIELDS = [BASE_MODELS_KEY, DATASETS_KEY, DISTANCE_LAMBDAS_KEY,
                  BATCH_SIZES_KEY, EMBEDDING_SIZES_KEY, VALIDATION_SIZES_KEY,
                  EPOCHS_KEY, BATCHNORM_KEY, DROPOUT_KEY, OPTIMIZERS_KEY,
                  LEARNING_RATES_KEY, TEST_ACCURACY_KEY, DURATION_KEY]
