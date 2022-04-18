import argparse
from distutils.log import debug
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
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Ransomware detection app")
    parser.add_argument("--csv_file_path", dest="csv_file_path", type=str, help="../data/detection_app/branch_instructions.csv")
    parser.add_argument("--model_path", dest="model_path", type=str, help="../checkpoints/BI-GAF/best-checkpoint.ckpt")
    parser.add_argument("--debug_mode", dest="debug_mode", type=bool, help="True or False", default=False)
    args = parser.parse_args()

    print("============================")
    print("  Ransomware Detection App  ")
    print("============================")
    print("Observation started...")

    observation_data = list()
    classifier: HpcClassifier = HpcClassifier(args.model_path)
    debug_counter: int = 0

    with open(args.csv_file_path) as file:
        while True:
            line = follow(file=file)
            if line:
                observation_data.append(line.split(","))
                if len(observation_data) == 50:
                    attack_detected: bool = classifier.classify(observation_data)
                    observation_data.clear()

                    if attack_detected:
                        time.sleep(5)
                    
                    # Just for debbuging
                    if debug_counter == 10:
                        break

                    if args.debug_mode:
                        debug_counter += 1
