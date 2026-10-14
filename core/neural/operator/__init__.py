"""Models for operator learning
"""
from core.neural.operator.fno     import FourierNeuralOperator, FNO
from core.neural.operator.fno_radial     import RadialFourierNeuralOperator, RadialFNO

# Aliases for modified names
ModifiedFNO:  type = FNO
ModifiedRFNO: type = RadialFNO
RFNO:         type = RadialFNO

from core.neural.collision_op.pfno  import PFNO
from core.neural.collision_op.prfno import PRFNO
from core.neural.collision_op.fno   import ConservativeFNO
from core.neural.collision_op.rfno  import ConservativeRFNO

__all__: list[str] = [
    'FourierNeuralOperator', 'FNO', 'ModifiedFNO',
    'RadialFourierNeuralOperator', 'RadialFNO', 'RFNO', 'ModifiedRFNO',
    'PFNO', 'PRFNO',
    'ConservativeFNO', 'ConservativeRFNO',
]


def main() -> None:
    pass


if __name__ == '__main__':
    main()
