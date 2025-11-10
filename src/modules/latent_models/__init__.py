REGISTRY = {}

from .old_mlp import TransitionModel as OldMLPModel
from .mlp import TransitionModel as MLPModel
from .inverse_mlp import InverseTransitionModel as InverseTransitionModel

REGISTRY["old_mlp"] = OldMLPModel
REGISTRY["mlp"] = MLPModel
REGISTRY["inverse_mlp"] = InverseTransitionModel
