"""planetar-fusion — Stage 1: a learned Sentinel-1 vessel detector benchmarked on xView3.

"Dark" is never a model output here: at deployment it is detection minus the
live AIS feed (docs/DESIGN.md §3, §6).
"""

__version__ = "0.1.0"
