from runners.ddpm import DDPM_RUNNER_CLASSES
from runners.edm import EDM_RUNNER_CLASSES
from runners.sde import SDE_RUNNER_CLASSES

_RUNNERS = {
    **DDPM_RUNNER_CLASSES,
    **EDM_RUNNER_CLASSES,
    **SDE_RUNNER_CLASSES,
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
