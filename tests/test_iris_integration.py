from ComputationalGraph.NeuralNetwork import NeuralNetwork


def test_iris_dataset_is_embedded_and_split_matches_cpp():
    # C++ uses 150 samples, shuffle, then 120 train / 30 test.
    nn = NeuralNetwork()

    train = []
    test = []
    nn.createIrisDataset(train, test, seed=1)

    assert len(train) == 120
    assert len(test) == 30

    # Each instance is 4 features + 1 label index => shape (5,)
    assert all(t.shape == (5,) for t in train)
    assert all(t.shape == (5,) for t in test)

    # Labels should be 0/1/2 only (C++ irisData has 3 classes)
    labels = {int(t.get((4,))) for t in train + test}
    assert labels.issubset({0, 1, 2})
    assert labels == {0, 1, 2}