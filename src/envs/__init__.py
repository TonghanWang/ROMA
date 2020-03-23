from functools import partial
#from smac.env import MultiAgentEnv, StarCraft2Env
from .multiagentenv import MultiAgentEnv
from .starcraft2 import StarCraft2Env
from .gfootball import GoogleFootballEnv

import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["gf"] = partial(env_fn, env=GoogleFootballEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
