from __future__ import annotations

from typing import List, Tuple
import random

from Math.Tensor import Tensor
from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.IrisData import IRIS_DATA


class NeuralNetwork(ComputationalGraph):
    def createIrisDataset(self, trainSet: List[Tensor], testSet: List[Tensor], seed: int = 1) -> None:
        rng = random.Random(seed)
        rows = [r[:] for r in IRIS_DATA]  # defensive copy
        rng.shuffle(rows)

        for i, r in enumerate(rows):
            # instance tensor shape (5,) : 4 features + label_index
            t = Tensor([float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])], (5,))
            if i < 120:
                trainSet.append(t)
            else:
                testSet.append(t)

    # ---- Iris instance helpers (mirrors C++ intent) ----
    @staticmethod
    def getLabelIndex(instance: Tensor) -> int:
        if len(instance.shape) != 1 or instance.shape[-1] < 2:
            raise ValueError(f"Expected a 1D instance tensor with feature(s)+label, got {instance.shape}")
        return int(instance.data[-1])

    @staticmethod
    def createInputTensor(instance: Tensor) -> Tensor:
        """
        C++ declares: Tensor createInputTensor(const Tensor& instance);
        Return features only, dropping the last entry which is the label.
        """
        if len(instance.shape) != 1 or instance.shape[-1] < 2:
            raise ValueError(f"Expected a 1D instance tensor with feature(s)+label, got {instance.shape}")
        size = int(instance.shape[-1]) - 1
        return Tensor([float(v) for v in instance.data[:size]], (size,))

    def test(self, testSet: List[Tensor]) -> float:
        count = 0
        total = 0
        for instance in testSet:
            self.inputNodes[0].setValue(self.createInputTensor(instance))
            output = self.predict()
            class_label = output[0]
            if class_label == self.getLabelIndex(instance):
                count += 1
            total += 1
        return (count + 0.0) / total if total > 0 else 0.0

    # ---- Output decoding (generic, used by tests + future models) ----
    def getClassLabels(self, outputNode) -> List[int]:
        """
        Convert output tensor to predicted class indices (argmax on last dim).
        Handles shapes like (C,), (1, C), (N, C), or higher-rank with last dim = C.
        """
        t: Tensor = outputNode.getValue()
        if t is None:
            raise ValueError("outputNode value is None")

        last = int(t.shape[-1])
        data = t.data

        # If Tensor stores nested lists for 2D+, some implementations still keep `data` flat.
        # Your Math.Tensor appears to store `data` in a Python list; rely on shape to slice.
        total = len(data)
        if total % last != 0:
            raise ValueError(f"Output data length {total} not divisible by class dim {last}")

        rows = total // last
        labels: List[int] = []
        for r in range(rows):
            start = r * last
            row = data[start : start + last]
            labels.append(max(range(last), key=lambda i: row[i]))
        return labels
