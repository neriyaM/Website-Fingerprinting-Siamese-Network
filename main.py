import csv

from loaders.runners import load_runners
import json
from consts import *


def main():
    with open(CONFIG_FILE_NAME) as f:
        config = json.load(f)
    runners = load_runners(config[RUNNERS_KEY])
    init_results_logger()

    for runner in runners:
        runner.run()
        runner.clear()


def init_results_logger():
    with open(RESULTS_FILE, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        writer.writeheader()


if __name__ == '__main__':
    main()
