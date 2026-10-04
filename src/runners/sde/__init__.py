from runners.sde.ou_oracle.runner import OUOracleRunner
from runners.sde.runner import CoupledDoubleWellLangevinRunner, SimpleOURunner

SDE_RUNNER_CLASSES = {
    cls.runner_name: cls
    for cls in (SimpleOURunner, CoupledDoubleWellLangevinRunner, OUOracleRunner)
}
