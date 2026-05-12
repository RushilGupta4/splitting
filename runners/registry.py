from runners.ddpm_gmm2d.runner import DDPMGMM2DRunner
from runners.edm_gmm2d.runner import EDMGMM2DRunner

_RUNNERS = {
    DDPMGMM2DRunner.runner_name: DDPMGMM2DRunner,
    EDMGMM2DRunner.runner_name: EDMGMM2DRunner,
}


def names():
    return tuple(sorted(_RUNNERS))


def get_runner_class(name: str):
    try:
        return _RUNNERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown runner '{name}'. Available: {', '.join(names())}"
        ) from exc
