from __future__ import annotations

from ComputationalGraph.NeuralNetwork import NeuralNetwork


def test_iris_embedded_dataset_split_and_extract():
    nn = NeuralNetwork()

    train, test = [], []
    nn.createIrisDataset(train, test, seed=1)

    assert len(train) == 120
    assert len(test) == 30

    # basic shape contract
    inst = train[0]
    assert inst.shape == (5,)

    x = nn.createInputTensor(inst)
    y = nn.getLabelIndex(inst)

    assert x.shape == (4,)
    assert y in (0, 1, 2)