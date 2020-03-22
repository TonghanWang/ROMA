REGISTRY = {}

from .rnn_agent import RNNAgent
from .latent_ce_dis_rnn_agent import LatentCEDisRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["latent_ce_dis_rnn"] = LatentCEDisRNNAgent
