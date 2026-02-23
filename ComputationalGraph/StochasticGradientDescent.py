from __future__ import annotations

from typing import Any, Dict, List

from Math.Tensor import Tensor
from ComputationalGraph.Optimizer import Optimizer, _tensor_sub_scaled


class StochasticGradientDescent(Optimizer):
    """
    C++: StochasticGradientDescent(learningRate, momentum)
    We'll implement momentum as classical velocity per-parameter tensor.
    """

    def __init__(self, learningRate: float, momentum: float):
        self.learningRate = float(learningRate)
        self.momentum = float(momentum)
        self._velocity: Dict[int, Tensor] = {}  # keyed by id(node)

    def updateValues(self, nodeMap: Dict[Any, List[Any]]) -> None:
        # Update all learnable nodes appearing as keys in nodeMap or in reverse edges.
        # Java likely updates learnable nodes globally; here we scan all nodes we can see.
        seen = set()

        def consider(node: Any) -> None:
            if node in seen:
                return
            seen.add(node)
            if not hasattr(node, "isLearnable") or not node.isLearnable():
                return
            v = node.getValue()
            g = node.getBackward()
            if v is None or g is None:
                return

            key = id(node)
            if key not in self._velocity:
                # initialize velocity with zeros
                self._velocity[key] = Tensor([0.0] * len(v.data), v.shape)

            vel = self._velocity[key]
            # vel = momentum*vel + grad
            for i in range(len(vel.data)):
                vel.data[i] = self.momentum * vel.data[i] + g.data[i]

            node.setValue(_tensor_sub_scaled(v, vel, self.learningRate))

        for parent, children in nodeMap.items():
            consider(parent)
            for ch in children:
                consider(ch)