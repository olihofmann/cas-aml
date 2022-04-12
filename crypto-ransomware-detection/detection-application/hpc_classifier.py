from matplotlib import rc_params_from_file
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
    IMAGE_FILE_NAME = "temp.jpg"

    def __init__(self, model_path: str) -> None:
        self.transformer: GramianAngularField = GramianAngularField()
        self.model: BaselineNetwork = BaselineNetwork(number_of_classes=2, image_size=50)
        self.model.load(model_path)
        self.temp_file = tempfile.NamedTemporaryFile()
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],)
        ])

    def _transform_to_image(self, df_transform: pd.DataFrame) -> np.ndarray:
        flatten_time_series: np.ndarray = np.array([df_transform.values.flatten()])
        return self.transformer.fit_transform(flatten_time_series)

    def _convert_to_dataframe(self, observation_data: list):
        df: pd.DataFrame = pd.DataFrame(observation_data)
        df = df.drop([2, 6, 7], axis=1)
        df = df.dropna(axis=1)
        df = df.reset_index()
        df.columns = self.DATASET_COLUMNS
        df["counter-value"] = pd.to_numeric(df["counter-value"])

        return df

    def classify(self, observation_data: list):
        df: pd.DataFrame = self._convert_to_dataframe(observation_data)

        df_counter_value: pd.DataFrame = df[["counter-value"]].copy()
        transformed_time_series: np.ndarray = self._transform_to_image(df_counter_value)

        plt.imsave(f"{self.temp_file.name}.jpg", transformed_time_series[0])
        image = skimage.io.imread(f"{self.temp_file.name}.jpg")
        image = self.transformation(image)

        prediction, confidence = self.model.predict(image.unsqueeze_(0))
        confidence_score: float = confidence.item()
        category: str = self.IDX_TO_CLASS[prediction.item()]

        if category == "ransomware" and confidence_score >= 0.7:
            print(Fore.RED, f"Ransomware attack, confidence score {confidence_score}")
        elif category == "ransomware" and confidence_score < 0.7:
            print(Fore.YELLOW, f"Ransomware attack or high compute work, confidence score {confidence_score}")
        else:
            print(Fore.GREEN, f"Normal work, confidence score {confidence_score}")

        print(Fore.RESET, f"HPC class: {category}, confidence: {confidence.item()}")
