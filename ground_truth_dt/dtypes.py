from dataclasses import dataclass
from sklearn.tree import DecisionTreeRegressor


@dataclass
class DecisionTreeResults:
    """Results for a single square's decision tree."""

    layer: int
    neuron: int
    tree: DecisionTreeRegressor
    train_R2: float
    train_MSE: float
    test_R2: float
    test_MSE: float