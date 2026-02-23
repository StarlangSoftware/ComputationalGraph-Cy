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
        if instance.shape[-1] != 5:
            raise ValueError(f"Expected iris instance shape (5,), got {instance.shape}")
        # last entry is label index
        return int(instance.data[4])

    @staticmethod
    def createInputTensor(instance: Tensor) -> Tensor:
        """
        C++ declares: Tensor createInputTensor(const Tensor& instance);
        Here: return features only (4 floats), drop the label.
        """
        if instance.shape[-1] != 5:
            raise ValueError(f"Expected iris instance shape (5,), got {instance.shape}")
        return Tensor([float(instance.data[0]), float(instance.data[1]), float(instance.data[2]), float(instance.data[3])], (4,))

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