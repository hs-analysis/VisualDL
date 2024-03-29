import torch
from torchmetrics import Metric


def get_metric_results(metrics: list) -> dict:
    """Wraps every metric in a dicionary.

    Args:
        metrics (list): The list of metrics

    Returns:
        dict: Containing the metric name as key and result as value
    """
    return {
        name: val
        for name, val in zip(
            [type(metric) for metric in metrics],
            [metric.compute() for metric in metrics],
        )
    }


class MyAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        # compute final result
        return self.correct.float() / self.total
