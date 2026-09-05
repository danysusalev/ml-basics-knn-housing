import numpy as np
import typing


class MinMaxScaler:
    def _init_():
        self.data_min = None
        self.data_max = None
        self.data_range = None

    def fit(self, data: np.ndarray) -> None:
        self.data_min = data.min(axis=0)
        self.data_max = data.max(axis=0)
        self.data_range = self.data_max - self.data_min

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.data_min) / self.data_range


class StandardScaler:
    def _init_():
        self.data_mean = None
        self.data_std = None

    def fit(self, data: np.ndarray) -> None:
        self.data_mean = data.mean(axis=0)
        self.data_std = data.std(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.data_mean) / self.data_std
