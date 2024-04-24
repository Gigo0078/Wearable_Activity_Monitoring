import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt

from .trace import Trace


class Recording:
    """
    A class for managing and interacting with a recording composed of multiple `Trace` objects.

    Attributes:
        filename (str): The name of the file from which the recording data was loaded.
        labels (dict): A dictionary of labels associated with the recording, loaded from the pickle file.
        data (dict): A dictionary where keys are strings identifying each `Trace` and values are `Trace` instances.

    Methods:
        plot(self, keys: Union[str, List[str], List[List[str]]], ylabels: Optional[Union[str, List[str]]] = None, labels: Optional[Union[str, List[str], List[List[str]]]] = None, title: Optional[str] = None):
            Plots the data for given keys with optional y-axis labels and data labels. This allows for visualizing
            the sensor data contained within the `Trace` instances part of this recording. The method supports plotting
            multiple keys in a single figure, with each key representing a different trace.

        to_dict(self) -> dict:
            Converts the recording data and labels into a dictionary format. This is useful for serialization or
            saving the state of the recording in a structured format. The returned dictionary includes labels and
            data, with each `Trace` instance's data converted to a dictionary.    

        save_to_pkl(self, path: str):
            Saves the recording data to a pickle file specified by the path. This method facilitates the persistence
            of the recording data, allowing it to be easily saved and loaded from the filesystem. The entire state of
            the recording, including labels and `Trace` data, is saved.

        __repr__(self) -> str:
            Provides a formatted string representation of the `Recording` instance. This includes the filename, labels,
            and a summary of the data contained within the recording. It is helpful for quickly understanding the
            contents and properties of a `Recording` object in a human-readable format.
    """

    def __init__(self, path: str):
        self.filename = os.path.basename(path)

        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
        self.labels = data_dict["labels"]
        self.data = {key: Trace.from_dict(trace_dict) for key, trace_dict in data_dict["data"].items()}  
  
    def plot(self, keys: Union[str, List[str], List[List[str]]], ylabels: Optional[Union[str, List[str]]] = None, labels: Optional[Union[str, List[str], List[List[str]]]] = None, title: Optional[str] = None, start_s: float = 0, end_s: float = float('inf')):

        if isinstance(keys, str):
            keys = [[keys]]
        elif (isinstance(keys, list) and not isinstance(keys[0], list)):
            keys = [keys]

        if ylabels is not None:
            if isinstance(ylabels, str):
                ylabels = [ylabels]
            assert len(ylabels) == len(keys), "Number of ylabels must match the number of key groups."
        
        if labels is not None:
            if isinstance(labels, str):
                labels = [[labels]]
            elif (isinstance(labels, list) and not isinstance(labels[0], list)):
                labels = [labels]
            assert len(labels) == len(keys), "Number of label groups must match the number of key groups."
            assert all([len(label_group) == len(key_group) for label_group, key_group in zip(labels, keys)]), "Number of labels must match the number of keys in each group."

        nrows = len(keys)
        fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 6), sharex=True, squeeze=False)

        for i, key_group in enumerate(keys):
            current_ax = axs[i, 0]
            current_ax.set_ylabel(ylabels[i] if ylabels else '')

            for j, key in enumerate(key_group):
                current_label = labels[i][j] if labels else key
                plot_x = self.data[key].timestamps
                plot_y = self.data[key].values

                idxs = np.where((plot_x > start_s) & (plot_x < end_s))[0]

                # current_ax.plot(self.data[key].timestamps, self.data[key].values, label=current_label)
                current_ax.plot(plot_x[idxs], plot_y[idxs], label=current_label)

            current_ax.legend(loc='lower right')
            current_ax.grid()

        axs[-1, 0].set_xlabel('Time[s]')

        if title:
            fig.suptitle(title, fontsize=16)  # Set the title for the figure

        plt.tight_layout()
        plt.show()

    def to_dict(self) -> dict:
        return {
            "labels": self.labels,
            "data": {key: dataset.to_dict() for key, dataset in self.data.items()}
        }

    def save_to_pkl(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.to_dict(), f)

    def __repr__(self) -> str:
        labels_str = str(self.labels) if self.labels else '{}'
        data_keys = sorted([f"{key}" for key in self.data.keys()])
        data = "{"+", ".join([f"'{k}': {self.data[k]}" for k in data_keys])+"}"
        return f"Recording(filename='{self.filename}', labels={labels_str}, data={data})"
