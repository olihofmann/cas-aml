import argparse
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
    parser.add_argument("--csv_file1_path", dest="csv_file1_path", type=str, help="../data/detection_app/branch_instructions.csv")
    parser.add_argument("--csv_file2_path", dest="csv_file2_path", type=str, help="../data/detection_app/branch_instructions.csv")
    parser.add_argument("--model_path", dest="model_path", type=str, help="../checkpoints/BI-GAF/best-checkpoint.ckpt")
    parser.add_argument("--debug_mode", dest="debug_mode", type=bool, help="True or False", default=False)
    args = parser.parse_args()

    print("============================")
    print("  Ransomware Detection App  ")
    print("============================")
    print("Observation started...")

    observation_data_hpc_1 = list()
    observation_data_hpc_2 = list()
    classifier: HpcClassifier = HpcClassifier(args.model_path)
    debug_counter: int = 0

    with open(args.csv_file1_path) as file1, open(args.csv_file2_path) as file2:
        while True:
            line1 = follow(file=file1)
            line2 = follow(file=file2)
            if line1 and line2:
                observation_data_hpc_1.append(line1.split(","))
                observation_data_hpc_2.append(line2.split(","))
                if len(observation_data_hpc_1) == 50 and len(observation_data_hpc_2):
                    attack_detected: bool = classifier.classify(observation_data_hpc_1, observation_data_hpc_2)
                    observation_data_hpc_1.clear()
                    observation_data_hpc_2.clear()
                    
                    # Just for debbuging
                    if debug_counter == 10:
                        break

                    if args.debug_mode:
                        debug_counter += 1
