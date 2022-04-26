import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import tempfile

from colorama import Fore
from pyts.image import GramianAngularField
from model import BaselineNetwork
from torchvision import transforms


class HpcClassifier:
    DATASET_COLUMNS = ["index", "time-interval", "counter-value", "event", "runtime", "percentage"]
    IDX_TO_CLASS = {0: "ransomware", 1: "benign"}

    def __init__(self, model_path: str) -> None:
        self.transformer: GramianAngularField = GramianAngularField()
        self.model: BaselineNetwork = BaselineNetwork(number_of_classes=2, image_size=50)
        self.model.load(model_path)
        self.temp_file_1 = tempfile.NamedTemporaryFile()
        self.temp_file_2 = tempfile.NamedTemporaryFile()
        self.ransomware_detection_counter: int = 0
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225],)
        ])

    def _transform_to_image(self, df_transform: pd.DataFrame) -> np.ndarray:
        flatten_time_series: np.ndarray = np.array([df_transform.values.flatten()])
        return self.transformer.fit_transform(flatten_time_series)

    def _convert_to_dataframe(self, observation_data: list):
        df: pd.DataFrame = pd.DataFrame(observation_data)
        df = df.drop([2, 6, 7], axis=1)
        df = df.reset_index()
        df.columns = self.DATASET_COLUMNS
        df["counter-value"] = pd.to_numeric(df["counter-value"])

        return df[["counter-value"]].copy()

    def classify(self, observation_data_1: list, observation_data_2: list) -> bool:
        try:
            df_counter_value_1: pd.DataFrame = self._convert_to_dataframe(observation_data_1)
            df_counter_value_2: pd.DataFrame = self._convert_to_dataframe(observation_data_2)

            transformed_time_series_1: np.ndarray = self._transform_to_image(df_counter_value_1)
            transformed_time_series_2: np.ndarray = self._transform_to_image(df_counter_value_2)

            filename_1: str = f"{self.temp_file_1.name}.jpg"
            filename_2: str = f"{self.temp_file_2.name}.jpg"
            plt.imsave(filename_1, transformed_time_series_1[0])
            plt.imsave(filename_2, transformed_time_series_2[0])

            image_1 = skimage.io.imread(filename_1)
            image_2 = skimage.io.imread(filename_2)

            image = np.concatenate((image_1, image_2), axis=2)
            image = self.transformation(image)

            prediction, confidence = self.model.predict(image.unsqueeze_(0))
            confidence_score: float = confidence.item()
            category: str = self.IDX_TO_CLASS[prediction.item()]

            if category == "ransomware" and confidence_score > 0.85:
                self.ransomware_detection_counter += 1

            attack_detected: bool = False
            if self.ransomware_detection_counter > 3:
                print(Fore.RED, "Potential Crypto-Ransomware attack!!!!!!!!!!!!!")
                attack_detected = True
            else:
                print(Fore.GREEN, "Normal work")

            return attack_detected
        except:
            return False
