
import time

from io import TextIOWrapper

from hpc_classifier import HpcClassifier


def follow(file: TextIOWrapper):
    where = file.tell()
    line = file.readline()
    if not line:
        time.sleep(1)
        file.seek(where)
        return None
    else:
        return line

if __name__ == "__main__":
    observation_data = list()
    classifier: HpcClassifier = HpcClassifier()

    with open("../data/detection_app/branch_instructions.csv") as file:
        while True:
            line = follow(file=file)
            if line:
                observation_data.append(line.split(","))
                if len(observation_data) == 50:
                    classifier.classify(observation_data)
                    observation_data.clear()
