from    typing      import  Optional, Sequence, Self
import  torch

from    sys         import  path    as  sys_path
from    pathlib     import  Path
sys_path.append(str(Path(__file__).parent))
from    deep_numerical.utils        import  velocity_grid, relative_error


##################################################
##################################################
__all__: list[str] = [
    'LossFunctions',
]


##################################################
##################################################
class LossFunctions():
    def __init__(
        self,
        dimension:      int,
        resolution:     int,
        v_max:          float,
        v_where_closed: str                     = 'none',
        dtype:          torch.dtype             = torch.float,
        device:         torch.device            = torch.device('cpu'),
    ) -> Self:
        """
        Initialize the loss functions for the Boltzmann equation.
        
        Arguments:
            `dimension` (`int`):
                The dimension of the velocity space (2 or 3).
            `resolution` (`int`):
                The resolution of the velocity grid.
            `v_max` (`float`):
                The maximum velocity.
            `v_where_closed` (`str`, default: 'none'):
                Where the velocity grid is closed ('none', 'periodic', 'dirichlet').
            `dtype` (`torch.dtype`, default: `torch.float`):
                The data type of the tensors.
            `device` (`torch.device`, default: `torch.device('cpu')`):
                The device to use for the tensors.
        """
        self.__dimension = dimension
        self.__resolution = resolution
        self.__v_max = v_max
        self.__v_where_closed = v_where_closed
        self.__dtype = dtype
        self.__device = device
        self.__v_axes = tuple(range(-1-dimension, -1))
        
        self.__v_grid = velocity_grid(dimension, resolution, v_max, where_closed=v_where_closed, dtype=dtype, device=device)
        self.__weight_mass: torch.Tensor = \
            torch.ones((*(resolution for _ in range(dimension)), 1), device=device)
        self.__weight_momentum: torch.Tensor = \
            self.__v_grid.clone()
        self.__weight_energy: torch.Tensor = \
            self.__v_grid.pow(2).sum(dim=-1, keepdim=True)
        
        return
    
    
    @property
    def dimension(self) -> int:
        return self.__dimension
    @property
    def resolution(self) -> int:
        return self.__resolution
    @property
    def v_max(self) -> float:
        return self.__v_max
    @property
    def v_where_closed(self) -> str:
        return self.__v_where_closed
    @property
    def dtype(self) -> torch.dtype:
        return self.__dtype
    @property
    def device(self) -> torch.device:
        return self.__device
    @property
    def v_axes(self) -> tuple[int, ...]:
        return self.__v_axes
    @property
    def v_grid(self) -> torch.Tensor:
        return self.__v_grid

    
    def compute_loss_data(
            self,
            preds:      Sequence[torch.Tensor],
            targets:    Sequence[torch.Tensor],
            p:          float   = 2.0,
            weights:    Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """
        Compute the data loss.
        
        Arguments:
            `preds` (`Sequence[torch.Tensor]`):
                A sequence of predicted tensors.
            `targets` (`Sequence[torch.Tensor]`):
                A sequence of target tensors.
            `p` (`float`, default: `2.0`):
                The power of the norm to be used for the relative error.
            `weights` (`Optional[torch.Tensor]`, default: `None`):
                A tensor of weights for each prediction-target pair. If `None`, uniform weights are used.
                
        Returns:
            `torch.Tensor`:
                The 1-dimensional tensor which saves the instance-wise loss terms.
        """
        assert len(preds)==len(targets)
        if weights is None:
            _shape = (preds[0].size(0), len(preds))
            weights = torch.ones(_shape, device=preds[0].device)
        else:
            assert weights.ndim==2
            assert weights.shape[1]==len(preds)
        
        loss_data = torch.zeros((1,), device=self.__device)
        for idx, (p_, t_) in enumerate(zip(preds, targets)):
            loss_data += (relative_error(p_, t_, p=p) * weights[:, idx]).sum()
        return loss_data
        
    
    def compute_loss_cons(
            self,
            preds:      Sequence[torch.Tensor],
            targets:    Sequence[torch.Tensor],
            p:          float   = 2.0,
            weights:    Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """
        Compute the loss regarding the conservation laws.
        
        Arguments:
            `preds` (`Sequence[torch.Tensor]`):
                A sequence of predicted tensors.
            `targets` (`Sequence[torch.Tensor]`):
                A sequence of target tensors.
            `p` (`float`, default: `2.0`):
                The power of the norm to be used for the relative error.
            `weights` (`Optional[torch.Tensor]`, default: `None`):
                A tensor of weights for each prediction-target pair. If `None`, uniform weights are used.
                
        Returns:
            `torch.Tensor`:
                The 1-dimensional tensor which saves the instance-wise loss terms.
        """
        assert len(preds)==len(targets)
        if weights is None:
            _shape = (preds[0].size(0), len(preds))
            weights = torch.ones(_shape, device=preds[0].device)
        else:
            assert weights.ndim==2
            assert weights.shape[1]==len(preds)
        
        loss_mass       = torch.zeros((1,), device=self.__device)
        loss_momentum   = torch.zeros((1,), device=self.__device)
        loss_energy     = torch.zeros((1,), device=self.__device)
        for idx, (p_, t_) in enumerate(zip(preds, targets)):
            loss_mass       += (
                relative_error(
                    (p_ * self.__weight_mass).mean(dim=self.__v_axes),
                    (t_ * self.__weight_mass).mean(dim=self.__v_axes),
                    p=p,
                ) * weights[:, idx]
            ).sum()
            loss_momentum   += (
                relative_error(
                    (p_ * self.__weight_momentum).mean(dim=self.__v_axes),
                    (t_ * self.__weight_momentum).mean(dim=self.__v_axes),
                    p=p,
                ) * weights[:, idx]
            ).sum()
            loss_energy     += (
                relative_error(
                    (p_ * self.__weight_energy).mean(dim=self.__v_axes),
                    (t_ * self.__weight_energy).mean(dim=self.__v_axes),
                    p=p,
                ) * weights[:, idx]
            ).sum()
        return loss_mass+loss_momentum+loss_energy


##################################################
##################################################
# End of file