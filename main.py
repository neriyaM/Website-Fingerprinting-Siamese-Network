import csv

from loader import load_data
from runner import load_runners
import json
from consts import *


def main():
    with open(CONFIG_FILE_NAME) as f:
        config = json.load(f)
    runners = load_runners(config[RUNNERS_KEY])
    X_train, Y_train = load_data("data/train/pairs/match.csv", "data/train/pairs/mismatch.csv",
                                 "data/train/features")
    X_test, Y_test = load_data("data/test/pairs/match.csv", "data/test/pairs/mismatch.csv",
                               "data/test/features")

    init_results_logger()

    for runner in runners:
        runner.run(X_train, Y_train, X_test, Y_test)
        runner.clear()


def init_results_logger():
    with open(RESULTS_FILE, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        writer.writeheader()


if __name__ == '__main__':
    main()
