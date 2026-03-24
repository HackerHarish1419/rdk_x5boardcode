import numpy as np
from hobot_dnn import pyeasy_dnn as dnn

class Infer:
    """
    Drop-in replacement for bpu_infer_lib_x5.Infer using hobot_dnn.pyeasy_dnn.

    Usage (same as original):
        model = Infer("helmet_best2.bin")
        outputs = model.infer([input_array])   # input_array: float32 NCHW
    """

    def __init__(self, model_path: str):
        self._models = dnn.load(model_path)
        self._model = self._models[0]

    def infer(self, inputs: list) -> list:
        """
        Run inference.

        Args:
            inputs: list of numpy arrays, one per model input (NCHW float32).

        Returns:
            list of numpy arrays, one per model output.
        """
        # pyeasy_dnn.Model.forward accepts a list of numpy arrays.
        outputs = self._model.forward(inputs)

        # Each element is a pyDNNTensor; extract .buffer (numpy array).
        return [out.buffer for out in outputs]
