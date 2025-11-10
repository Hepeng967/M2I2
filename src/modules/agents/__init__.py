REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .full_comm_agent import FullCommAgent
from .M2I2_agent import M2I2Agent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["full_comm"] = FullCommAgent
REGISTRY["M2I2"] = M2I2Agent