REGISTRY = {}

from .basic_controller import BasicMAC
from .separate_controller import SeparateMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["separate_mac"]=SeparateMAC