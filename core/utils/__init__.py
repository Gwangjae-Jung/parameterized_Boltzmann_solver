"""Utility functions and modules for core.
"""
from typing import Any, List, Union
import torch

from core.utils.dtype        import *
from core.utils.grid         import *
from core.utils.metric       import *
from core.utils.time_utils   import *
from core.utils._normalizers import *

from core.utils.dtype        import __all__ as __all_dtype
from core.utils.grid         import __all__ as __all_grid
from core.utils.metric       import __all__ as __all_metric
from core.utils.time_utils   import __all__ as __all_time
from core.utils._normalizers import __all__ as __all_norm


def count_parameters(models: Any, complex_as_two: bool = True) -> Union[int, List[int]]:
    """Counts the number of learnable parameters in one or more PyTorch models.

    ## Description
    Delegates to `core.neural.utils.count_parameters`.

    ## Arguments
    `models` (`Any`): A model or collection of models whose parameters are counted.
    `complex_as_two` (`bool`, default: `True`): If `True`, complex parameters are counted as two real parameters.

    ## Returns
    `Union[int, List[int]]`: The parameter count of the model.
    """
    from core.neural.utils.model_construction import count_parameters as _count
    return _count(models, complex_as_two=complex_as_two)


def initialize_weights(
        models:      Any,
        init_name:   str = "xavier normal",
        init_kwargs: dict[str, object] = {},
    ) -> None:
    """Initializes weights in PyTorch model(s).

    ## Description
    Delegates to `core.neural.utils.initialize_weights`.

    ## Arguments
    `models` (`Any`): A single model or collection of models to initialize.
    `init_name` (`str`, default: `"xavier normal"`): The initialization method name.
    `init_kwargs` (`dict[str, object]`, default: `{}`): Additional keyword arguments.

    ## Returns
    `None`: Initializes model parameters in-place.
    """
    from core.neural.utils.model_construction import initialize_weights as _init
    return _init(models, init_name=init_name, init_kwargs=init_kwargs)


__all__: list[str] = (
    list(__all_dtype)
    + list(__all_grid)
    + list(__all_metric)
    + list(__all_time)
    + list(__all_norm)
    + ['count_parameters', 'initialize_weights']
)


def main() -> None:
    pass


if __name__ == '__main__':
    main()
