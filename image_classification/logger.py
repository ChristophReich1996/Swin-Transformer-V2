from typing import Any, Dict

import torch
import os
from datetime import datetime


class Logger(object):
    """
    Class to log different metrics.
    """

    def __init__(self,
                 experiment_path: str = os.path.join(os.getcwd(), "experiments",
                                                     datetime.now().strftime("%d_%m_%Y__%H_%M_%S")),
                 experiment_path_extension: str = "",
                 path_metrics: str = "metrics",
                 path_models: str = "models") -> None:
        """
        Constructor method
        :param experiment_path: (str) Path to experiment folder
        :param path_metrics: (str) Path to folder in which all metrics are stored
        :param experiment_path_extension: (str) Extension to experiment folder
        :param path_models: (str)  Path to folder in which all models are stored
        """
        experiment_path = experiment_path + experiment_path_extension
        # Save parameters
        self.path_metrics = os.path.join(experiment_path, path_metrics)
        self.path_models = os.path.join(experiment_path, path_models)
        # Init folders
        os.makedirs(self.path_metrics, exist_ok=True)
        os.makedirs(self.path_models, exist_ok=True)
        # Init dicts to store the metrics and hyperparameters
        self.metrics = dict()

    def log_metric(self,
                   metric_name: str,
                   value: Any) -> None:
        """
        Method writes a given metric value into a dict including list for every metric.
        :param metric_name: (str) Name of the metric
        :param value: (float) Value of the metric
        """
        if metric_name in self.metrics:
            self.metrics[metric_name].append(float(value))
        else:
            self.metrics[metric_name] = [float(value)]

    def save_model(self,
                   model_sate_dict: Dict,
                   name: str) -> None:
        """
        Saves a given state dict
        :param model_sate_dict: (Dict) State dict to be saved
        :param name: (str) Name of the file
        """
        torch.save(obj=model_sate_dict, f=os.path.join(self.path_models, name + ".pt"))

    def save(self) -> None:
        """
        Method saves all current logs (metrics and hyperparameters). Plots are saved directly.
        """
        # Iterate items in metrics dict
        for metric_name, values in self.metrics.items():
            # Convert list of values to torch tensor to use build in save method from torch
            values = torch.tensor(values)
            # Save values
            torch.save(values, os.path.join(self.path_metrics, '{}.pt'.format(metric_name)))
