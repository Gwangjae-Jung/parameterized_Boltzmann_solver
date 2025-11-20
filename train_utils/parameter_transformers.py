from    typing      import  Sequence
import  torch
import  append_path


##################################################
##################################################
__all__: list[str] = [
    "params_as_constant",
    "params_as_power",
    "params_as_itself",
    "params_as_combination",
    "func01",
    "reconstruct_kernel",
]


##################################################
##################################################
def params_as_constant(
        params:             torch.Tensor,
        dimension:          int,
        resolution:         int,
    ) -> torch.Tensor:
    params = params.reshape(-1, *(1 for _ in range(dimension+1)))
    return params.repeat(1, *(resolution for _ in range(dimension)), 1)


def params_as_power(
        params:             torch.Tensor,
        dimension:          int,
        resolution:         int,
        v_max:              torch.Tensor,
    ) -> torch.Tensor:
    from    deep_numerical.utils    import  velocity_grid
    s_grid = velocity_grid(
        dimension, resolution, v_max, where_closed='none',
        dtype   = params.dtype,
        device  = params.device,
    ).norm(p=2, dim=-1, keepdim=True)
    s_grid = s_grid[None, ...]  # Add batch dimension
    params = params.reshape(-1, *(1 for _ in range(dimension+1)))
    return s_grid.pow(params)
    

##################################################
##################################################
def params_as_itself(params: torch.Tensor) -> torch.Tensor:
    return params


def params_as_combination(params: torch.Tensor, v_max: float, res_params: int) -> torch.Tensor:
    assert params.ndim==2 and params.shape[1]==1
    radii: torch.Tensor
    radii   = torch.reshape(torch.linspace(0, v_max, res_params, dtype=params.dtype, device=params.device), (1, -1))
    params  = params.repeat((1, res_params))
    linear  = radii * params
    power   = radii.pow(params)
    return torch.cat((params, linear, power), dim=1)


func01 = params_as_combination
reconstruct_kernel = params_as_power


##################################################
##################################################
# End of file