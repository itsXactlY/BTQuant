import numpy as np

from neural_trading_system.data.label_generator import LabelGenerator


def test_labels_shapes_and_bounds():
    prices = np.linspace(100, 130, 300)
    labels = LabelGenerator().generate(prices)
    assert labels["mfe"].shape == prices.shape
    assert labels["mae"].shape == prices.shape
    assert (labels["tp_prob"] >= 0).all() and (labels["tp_prob"] <= 1).all()
