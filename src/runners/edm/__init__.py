from runners.edm.gmm2d.runner import EDMGMM2DRunner
from runners.edm.runner import EDMRunner

EDM_RUNNER_CLASSES = {
    EDMGMM2DRunner.runner_name: EDMGMM2DRunner,
}

__all__ = ["EDMGMM2DRunner", "EDMRunner", "EDM_RUNNER_CLASSES"]
