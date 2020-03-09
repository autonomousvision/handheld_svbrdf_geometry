import torch
# these try except clauses are to handle cases where the hardware doesn't 
# support the functions, but you don't want to use them anyway
try:
    from TOME._implementation import depth_reprojection
except ImportError as e:
    def depth_reprojection(*args, **kwargs):
        raise e
try:
    from TOME._implementation import depth_reprojection_bound
except ImportError as e:
    def depth_reprojection_bound(*args, **kwargs):
        raise e

try:
    from TOME._implementation import permutohedral_filter
except ImportError as e:
    def permutohedral_filter(input, positions, reverse):
        raise e
