from    typing      import  Sequence
import  torch
import  append_path


##################################################
##################################################
__all__: list[str] = [
    "params_as_constant",
    "params_as_power_const",
    "params_as_itselt",
    "params_as_combination",
    "func01",
    "reconstruct_kernel",
]


##################################################
##################################################
def params_as_constant(
        alpha:          torch.Tensor,
        beta:           torch.Tensor,
        dimension:      int,
        resolution:     int,
    ) -> torch.Tensor:
    newshape_base = (-1, *(1 for _ in range(dimension+1)))
    alpha, beta = map(lambda x: torch.reshape(x, newshape_base), (alpha, beta))
    alpha   = alpha.repeat(1, *(resolution for _ in range(dimension)), 1)
    beta    = beta.repeat(1, *(resolution for _ in range(dimension)), 1)
    return torch.cat((alpha, beta), dim=-1)


def params_as_power_const(
        alpha:          torch.Tensor,
        beta:           torch.Tensor,
        dimension:      int,
        resolution:     int,
        v_max:          torch.Tensor,
    ) -> torch.Tensor:
    from    deep_numerical.utils    import  velocity_grid
    s_grid = velocity_grid(
        dimension, resolution, v_max, where_closed='none',
        dtype   = alpha.dtype,
        device  = alpha.device,
    ).norm(p=2, dim=-1, keepdim=True)
    s_grid  = s_grid[None, ...]  # Add batch dimension
    alpha   = alpha.reshape(-1, *(1 for _ in range(dimension+1)))
    beta    = beta.reshape(-1, *(1 for _ in range(dimension+1)))
    beta    = beta.repeat(1, *(resolution for _ in range(dimension)), 1)
    return torch.cat((s_grid.pow(alpha), beta), dim=-1)
    

##################################################
##################################################
def params_as_itselt(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return torch.cat((alpha, beta), dim=-1)


def params_as_combination(alpha: torch.Tensor, beta: torch.Tensor, v_max: float, res_params: int) -> torch.Tensor:
    assert alpha.ndim==2 and alpha.shape[1]==1
    assert beta.ndim==2 and beta.shape[1]==1
    radial  = torch.linspace(0, v_max, res_params, dtype=alpha.dtype, device=alpha.device)
    angular = torch.linspace(0, torch.pi, res_params, dtype=alpha.dtype, device=alpha.device)
    radial  = torch.reshape(radial, (1, -1))
    angular = torch.reshape(angular, (1, -1)).cos() + 1
    radial  = radial.pow(alpha)
    angular = angular.pow(beta)
    return torch.cat((radial, angular), dim=1)


func01 = params_as_combination
reconstruct_kernel = params_as_power_const


##################################################
##################################################
# End of file