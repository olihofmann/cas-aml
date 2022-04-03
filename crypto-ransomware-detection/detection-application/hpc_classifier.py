import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid

from pyts.image import GramianAngularField
from PIL import Image
from model import BaselineNetwork
from torchvision import transforms

class HpcClassifier:
    DATASET_COLUMNS = ["time-interval", "counter-value", "event", "runtime", "percentage"]
    IDX_TO_CLASS = {0: "ransomware", 1: "benign"}

    def __init__(self) -> None:
        self.transformer: GramianAngularField = GramianAngularField()
        self.model: BaselineNetwork = BaselineNetwork(number_of_classes=2, image_size=50)
        self.model.load("../checkpoints/BI-GAF/best-checkpoint.ckpt")

    def _transform_to_image(self, df_transform: pd.DataFrame) -> np.ndarray:
        flatten_time_series: np.ndarray = np.array([df_transform.values.flatten()])
        return self.transformer.fit_transform(flatten_time_series)

    def classify(self, obersation_data: list):
        df: pd.DataFrame = pd.DataFrame(obersation_data, columns=self.DATASET_COLUMNS)
        df["counter-value"] = pd.to_numeric(df["counter-value"])

        df_counter_value: pd.DataFrame = df[["counter-value"]].copy()
        transformed_time_series: np.ndarray = self._transform_to_image(df_counter_value)
        image_time_series = Image.fromarray(transformed_time_series[0], mode="RGB")
        plt.imsave(f"{uuid.uuid4()}.png", image_time_series)
        
        image_tensor = transforms.ToTensor()(image_time_series).unsqueeze_(0)
        prediction = self.model.predict(image_tensor)
        print(self.IDX_TO_CLASS[prediction.item()])
