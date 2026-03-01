from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import pickle

from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode
from ComputationalGraph.Node.ConcatenatedNode import ConcatenatedNode

from Math.Tensor import Tensor


class ComputationalGraph:
    """
    Python port of Java ComputationalGraph/ComputationalGraph.java.

    Semantics to preserve:
    - nodeMap: parent -> [children]
    - reverseNodeMap: child -> [parents]
    - topologicalSort(): DFS postorder; first element ends up being output node.
      forwardCalculation consumes nodes from END (inputs toward output).
      backpropagation consumes nodes from FRONT (output toward inputs).
    """

    def __init__(self) -> None:
        self.nodeMap: Dict[Any, List[Any]] = {}
        self.reverseNodeMap: Dict[Any, List[Any]] = {}
        self.inputNodes: List[Any] = []

    # --- abstract surface (mirrors Java) ---
    def train(self, trainSet: List[Tensor], parameters: Any) -> None:
        raise NotImplementedError

    def test(self, testSet: List[Tensor]) -> Any:
        raise NotImplementedError

    def getClassLabels(self, outputNode: Any) -> List[int]:
        raise NotImplementedError

    # --- graph wiring ---
    def _link(self, parent: Any, child: Any) -> None:
        self.nodeMap.setdefault(parent, []).append(child)
        self.reverseNodeMap.setdefault(child, []).append(parent)

    def addEdge(self, first: Any, second: Any, isBiased: bool) -> Any:
        """
        Java: addEdge(ComputationalNode first, Object second, boolean isBiased)

        second can be:
        - Function (has calculate/derivative)  -> creates a ComputationalNode(function=second)
        - Node (weights / another node)       -> creates a MultiplicationNode and links (first, second) -> newNode
        - MultiplicationNode (rare as "second", but allow it for parity) -> same as Node, but keep hadamard flag
        """
        # Case 1: Function edge
        if hasattr(second, "calculate") and hasattr(second, "derivative"):
            newNode = self._new_computational_node(learnable=False, function=second, isBiased=isBiased)
            self._link(first, newNode)
            return newNode

        # Case 2: Node-like (including MultiplicationNode weights node)
        if self._is_node(second):
            is_hadamard = bool(second.isHadamard()) if hasattr(second, "isHadamard") else False
            newNode = self._new_multiplication_node(
                learnable=False,
                isBiased=isBiased,
                isHadamard=is_hadamard,
                priorityNode=first,
            )
            self._link(first, newNode)
            self._link(second, newNode)
            return newNode

        raise ValueError("Illegal Type of Object: second")

    def addEdgeMul(self, first: Any, second: Any, isBiased: bool, isHadamard: bool) -> Any:
        """Java: addEdge(first, second, isBiased, isHadamard)."""
        newNode = self._new_multiplication_node(
            learnable=False, isBiased=isBiased, isHadamard=isHadamard, priorityNode=first
        )
        self._link(first, newNode)
        self._link(second, newNode)
        return newNode

    def addAdditionEdge(self, first: Any, second: Any, isBiased: bool) -> Any:
        """Java: addAdditionEdge(...) creates a plain node with no function."""
        newNode = self._new_computational_node(learnable=False, function=None, isBiased=isBiased)
        self._link(first, newNode)
        self._link(second, newNode)
        return newNode

    def concatEdges(self, nodes: List[Any], dimension: int) -> Any:
        """Java: creates ConcatenatedNode(dimension) and links all inputs to it."""
        newNode = self._new_concatenated_node(dimension=dimension)
        for n in nodes:
            self._link(n, newNode)
            if hasattr(newNode, "addNode"):
                newNode.addNode(n)
        return newNode

    # --- topo sort ---
    def _sortRecursive(self, node: Any, visited: Set[Any]) -> List[Any]:
        queue: List[Any] = []
        visited.add(node)
        if node in self.nodeMap:
            for child in self.nodeMap[node]:
                if child not in visited:
                    queue.extend(self._sortRecursive(child, visited))
        queue.append(node)  # postorder
        return queue

    def _find_output_node(self, nodes: List[Any]) -> Any:
        # sink = node with no outgoing edges
        sinks = [n for n in nodes if (n not in self.nodeMap) or (len(self.nodeMap.get(n, [])) == 0)]
        return sinks[0] if sinks else nodes[0]

    
    def topologicalSort(self) -> List[Any]:
        visited: Set[Any] = set()
        sortedList: List[Any] = []

        # include keys + children so sinks are also included
        all_nodes: Set[Any] = set(self.nodeMap.keys())
        for children in self.nodeMap.values():
            all_nodes.update(children)

        for node in all_nodes:
            if node not in visited:
                sortedList.extend(self._sortRecursive(node, visited))

        return sortedList

    # --- clear ---
    def _clearRecursive(self, visited: Set[Any], node: Any) -> None:
        visited.add(node)
        if hasattr(node, "isLearnable") and not node.isLearnable():
            if hasattr(node, "setValue"):
                node.setValue(None)
        if hasattr(node, "setBackward"):
            node.setBackward(None)
        if node in self.nodeMap:
            for child in self.nodeMap[node]:
                if child not in visited:
                    self._clearRecursive(visited, child)

    def clear(self) -> None:
        visited: Set[Any] = set()
        for node in list(self.nodeMap.keys()):
            if node not in visited:
                self._clearRecursive(visited, node)

    # --- tensor helpers matching Java semantics ---
    @staticmethod
    def _transposeAxes(length: int) -> Tuple[int, ...]:
        axes = list(range(length))
        if length >= 2:
            axes[-1], axes[-2] = axes[-2], axes[-1]
        return tuple(axes)

    @staticmethod
    def _getBiasedPartial(t: Tensor) -> Tensor:
        end = list(t.shape)
        end[-1] = end[-1] - 1
        start = (0,) * len(t.shape)
        return t.partial(start, tuple(end))

    @staticmethod
    def _matmul(a: Tensor, b: Tensor) -> Tensor:
        # Java Tensor.multiply() == matmul; Python Tensor.dot provides matmul
        return a.dot(b)

    @staticmethod
    def _hadamard(a: Tensor, b: Tensor) -> Tensor:
        # Java hadamardProduct
        return a * b

    @staticmethod
    def _concat(a: Tensor, b: Tensor, dim: int) -> Tensor:
        # Tensor API doesn't expose concat; implement using get/set.
        if len(a.shape) != len(b.shape):
            raise ValueError("concat requires same rank")
        for i in range(len(a.shape)):
            if i != dim and a.shape[i] != b.shape[i]:
                raise ValueError("concat requires equal shapes except concat dim")

        new_shape = list(a.shape)
        new_shape[dim] = a.shape[dim] + b.shape[dim]
        out = Tensor([0.0] * _numel(tuple(new_shape)), tuple(new_shape))

        def iter_indices(shape: Tuple[int, ...]):
            strides = _strides(shape)
            for flat in range(_numel(shape)):
                yield tuple(_unflatten(flat, strides))

        for idx in iter_indices(a.shape):
            out.set(idx, a.get(idx))
        for idx in iter_indices(b.shape):
            out_idx = list(idx)
            out_idx[dim] += a.shape[dim]
            out.set(tuple(out_idx), b.get(idx))
        return out

    # --- backprop core ---
    def _calculateDerivative(self, node: Any, child: Any) -> Optional[Tensor]:
        reverseParents = self.reverseNodeMap.get(child)
        if not reverseParents:
            return None

        child_backward: Optional[Tensor] = child.getBackward() if hasattr(child, "getBackward") else None
        if child_backward is None:
            return None

        if hasattr(child, "isBiased") and child.isBiased():
            backward = self._getBiasedPartial(child_backward)
        else:
            backward = child_backward

        func = child.getFunction() if hasattr(child, "getFunction") else None
        if func is not None:
            child_value: Optional[Tensor] = child.getValue()
            if child_value is None:
                return None
            if hasattr(child, "isBiased") and child.isBiased():
                child_value = self._getBiasedPartial(child_value)
            return func.derivative(child_value, backward)

        if self._is_concatenated_node(child):
            dim = child.getDimension()
            idx_in_concat = child.getIndex(node)
            parents = reverseParents
            block = backward.shape[dim] // len(parents)

            new_shape = list(backward.shape)
            new_shape[dim] = block
            out = Tensor([0.0] * _numel(tuple(new_shape)), tuple(new_shape))

            out_strides = _strides(tuple(new_shape))
            for flat in range(_numel(tuple(new_shape))):
                out_idx = _unflatten(flat, out_strides)
                bwd_idx = list(out_idx)
                bwd_idx[dim] += idx_in_concat * block
                out.set(tuple(out_idx), backward.get(tuple(bwd_idx)))
            return out

        if self._is_multiplication_node(child):
            left, right = reverseParents[0], reverseParents[1]
            is_hadamard = child.isHadamard()
            if left == node:
                right_val = right.getValue()
                if is_hadamard:
                    return self._hadamard(right_val, backward)
                rt = right_val.transpose(self._transposeAxes(len(right_val.shape)))
                return self._matmul(backward, rt)
            else:
                left_val = left.getValue()
                if is_hadamard:
                    return self._hadamard(left_val, backward)
                lt = left_val.transpose(self._transposeAxes(len(left_val.shape)))
                return self._matmul(lt, backward)

        return backward

    def _calculateRMinusY(self, outputNode: Any, classLabelIndex: List[int]) -> None:
        out_val: Tensor = outputNode.getValue()
        last_dim = out_val.shape[-1]
        values: List[float] = []
        for i, ov in enumerate(out_val.data):
            if (i % last_dim) == classLabelIndex[i // last_dim]:
                values.append(1.0 - ov)
            else:
                values.append(-ov)
        outputNode.setBackward(Tensor(values, out_val.shape))

    def backpropagation(self, optimizer: Any, classLabelIndex: List[int]) -> None:
        sortedNodes = self.topologicalSort()
        if not sortedNodes:
            return

        outputNode = self._find_output_node(sortedNodes)
        self._calculateRMinusY(outputNode, classLabelIndex)

        if sortedNodes:
            sortedNodes.pop(0).setBackward(outputNode.getBackward())

        while sortedNodes:
            node = sortedNodes.pop(0)
            for child in self.nodeMap.get(node, []):
                deriv = self._calculateDerivative(node, child)
                if deriv is None:
                    continue
                if node.getBackward() is None:
                    node.setBackward(deriv)
                else:
                    node.setBackward(node.getBackward() + deriv)

        optimizer.updateValues(self.nodeMap)
        self.clear()

    # --- bias handling (matches Java getBiased) ---
    @staticmethod
    def _biasTensorValue(t: Tensor) -> Tensor:
        # Java getBiased(): append 1.0 after each row (last-dimension block).
        # For 1D, treat as a single row (1, D).
        if len(t.shape) == 1:
            t = t.reshape((1, t.shape[0]))

        rows = 1
        for d in t.shape[:-1]:
            rows *= d
        cols = t.shape[-1]
        new_shape = tuple(t.shape[:-1]) + (cols + 1,)

        out = Tensor([0.0] * _numel(new_shape), new_shape)

        # Copy row by row and set last col to 1.0
        out_strides = _strides(new_shape)
        for flat in range(_numel(new_shape)):
            idx = _unflatten(flat, out_strides)
            if idx[-1] == cols:
                out.set(tuple(idx), 1.0)
            else:
                out.set(tuple(idx), t.get(tuple(idx)))
        return out

    def predict(self) -> List[int]:
        labels = self.forwardCalculation(enableDropout=False)
        self.clear()
        return labels

    def forwardCalculationTrain(self) -> List[int]:
        return self.forwardCalculation(enableDropout=True)

    def forwardCalculation(self, enableDropout: bool = True) -> List[int]:
        sortedNodes = self.topologicalSort()
        if not sortedNodes:
            return []

        outputNode = self._find_output_node(sortedNodes)
                

        # Output node = sink (no outgoing edges). This is robust regardless of topo ordering.
        sinks = [n for n in sortedNodes if (n not in self.nodeMap) or (len(self.nodeMap.get(n, [])) == 0)]
        outputNode = sinks[0] if sinks else sortedNodes[0]

        concatenatedNodeMap: Dict[Any, List[Optional[Any]]] = {}
        counterMap: Dict[Any, int] = {}

        # Process from inputs -> output by iterating reverse topo, skipping the sink itself.
        for current in reversed(sortedNodes):
            if current is outputNode:
                continue

            if current.isBiased():
                v = current.getValue()
                if v is None:
                    raise ValueError("Current node's value is null")
                current.setValue(self._biasTensorValue(v))

            if current.getValue() is None:
                raise ValueError("Current node's value is null")

            for child in self.nodeMap.get(current, []):
                if child.getValue() is None:
                    func = child.getFunction()
                    if func is not None:
                        currentValue = current.getValue()
                        if self._is_dropout(func):
                            if enableDropout:
                                child.setValue(func.calculate(currentValue))
                            else:
                                child.setValue(Tensor(list(currentValue.data), currentValue.shape))
                        else:
                            child.setValue(func.calculate(currentValue))
                    else:
                        if self._is_concatenated_node(child):
                            parents = self.reverseNodeMap.get(child, [])
                            if child not in concatenatedNodeMap:
                                concatenatedNodeMap[child] = [None] * len(parents)

                            idx = child.getIndex(current)
                            if concatenatedNodeMap[child][idx] is None:
                                counterMap[child] = counterMap.get(child, 0) + 1
                            concatenatedNodeMap[child][idx] = current

                            if counterMap.get(child, 0) == len(parents) and all(x is not None for x in concatenatedNodeMap[child]):
                                base = concatenatedNodeMap[child][0].getValue()
                                for i in range(1, len(concatenatedNodeMap[child])):
                                    base = self._concat(base, concatenatedNodeMap[child][i].getValue(), child.getDimension())
                                child.setValue(base)
                        else:
                            child.setValue(current.getValue())
                else:
                    if self._is_multiplication_node(child):
                        childValue = child.getValue()
                        currentValue = current.getValue()
                        if child.isHadamard():
                            child.setValue(self._hadamard(childValue, currentValue))
                        elif child.getPriorityNode() != current:
                            child.setValue(self._matmul(childValue, currentValue))
                        else:
                            child.setValue(self._matmul(currentValue, childValue))
                    else:
                        child.setValue(child.getValue() + current.getValue())

        return self.getClassLabels(outputNode)

    # --- persistence (Java save/loadModel) ---
    def save(self, fileName: str) -> None:
        try:
            with open(fileName, "wb") as f:
                pickle.dump(self, f)
        except OSError:
            print("Object could not be saved.")

    @staticmethod
    def loadModel(fileName: str) -> Optional["ComputationalGraph"]:
        try:
            with open(fileName, "rb") as f:
                return pickle.load(f)
        except (OSError, pickle.UnpicklingError):
            return None

    def _new_computational_node(self, learnable: bool, function: Any, isBiased: bool) -> Any:
        # operator=None corresponds to Java "function node" or "addition node" (no function)
        return ComputationalNode(learnable=learnable, function=function, isBiased=isBiased)

    def _new_multiplication_node(self, learnable: bool, isBiased: bool, isHadamard: bool, priorityNode: Any) -> Any:
        return MultiplicationNode(
            learnable=learnable,
            isBiased=isBiased,
            isHadamard=isHadamard,
            priorityNode=priorityNode,
        )

    def _new_concatenated_node(self, dimension: int) -> Any:
        return ConcatenatedNode(dimension)

    def _is_node(self, x: Any) -> bool:
        return hasattr(x, "getValue") and hasattr(x, "setValue")

    def _is_multiplication_node(self, x: Any) -> bool:
        return isinstance(x, MultiplicationNode)

    def _is_concatenated_node(self, x: Any) -> bool:
        return isinstance(x, ConcatenatedNode)

    @staticmethod
    def _is_dropout(func: Any) -> bool:
        return func.__class__.__name__ == "Dropout"


def _numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    strides: List[int] = []
    prod = 1
    for d in reversed(shape):
        strides.append(prod)
        prod *= int(d)
    return tuple(reversed(strides))


def _unflatten(flat_index: int, strides: Tuple[int, ...]) -> List[int]:
    idx: List[int] = []
    for s in strides:
        idx.append(flat_index // s)
        flat_index %= s
    return idx
