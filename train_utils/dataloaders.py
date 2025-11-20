from    typing      import  Optional, Sequence, Union, Self
from    pathlib     import  Path
import  torch


##################################################
##################################################
__all__: list[str] = [
    'load_data', 'augment_data_2D', 'augment_data_3D', "WeightedDataset",
]


##################################################
##################################################
def load_data(
        directory:  Union[str, Path],
        resolution: int,
        alpha:      Sequence[float],
        part_index: Sequence[int],
        dtype:      torch.dtype = torch.float,
        size_zfill: int = 2,
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """
    Arguments:
        `directory` (`Union[str, Path]`):
            The directory where the data files are stored.
        `resolution` (`int`):
            The resolution of the data.
        `alpha` (`Sequence[float]`):
            The values of alpha of the data to be loaded.
        `part_index` (`Sequence[int]`):
            The parts of the data to be loaded.
        `dtype` (`torch.dtype`, default: `torch.float`):
            The data type of the loaded tensors.
        `size_zfill` (`int`, default: `2`):
            The number of digits to pad the resolution and part index in the file names.
    
    Returns:
        `dict[str, torch.Tensor]`:
            A dictionary containing the concatenated `data`, `vhs_coeff`, and `vhs_alpha`.
        `dict[str, object]`:
            A dictionary containing shared attributes such as `v_max` and `where_closed`.
    """
    from    itertools       import  product
    
    if not isinstance(directory, Path):
        directory = Path(directory)

    def _get_file_name(res: int, a: float, p: int) -> str:
        res = str(res).zfill(3)
        p   = str(p).zfill(size_zfill)
        return f"res{res}__alpha{a:.1e}__part{p}.pth"
        
    files = [
        torch.load(
            directory/_get_file_name(resolution, a, p),
            weights_only = False,
            map_location = torch.device('cpu'),
        )
        for a, p in product(alpha, part_index)
    ]
    
    data: torch.Tensor = torch.cat([f['data'].type(dtype) for f in files], dim=0)
    ndim = (data.ndim-3)//2
    data = data[:, :, *(0 for _ in range(ndim)), ...]   # Remove the space dimensions
    vhs_coeff: torch.Tensor = \
        torch.cat(
            [
                float(f['vhs_coeff']) * torch.ones((f['n_init'], 1), dtype=dtype)
                for f in files
            ],
            dim=0,
        )
    vhs_alpha: torch.Tensor = \
        torch.cat(
            [
                float(f['vhs_alpha']) * torch.ones((f['n_init'], 1), dtype=dtype)
                for f in files
            ],
            dim=0,
        )
    elapsed_time: float = sum(f['elapsed_time'] for f in files)

    return (
        {
            'data':         data,
            'vhs_coeff':    vhs_coeff,
            'vhs_alpha':    vhs_alpha,
        },
        {
            'v_max':            float(files[0]['v_max']),
            'v_where_closed':   str(files[0]['v_where_closed']),
            'elapsed_time':     elapsed_time,
        }
    )


def augment_data_2D(data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Augment the data for 2D by flipping and rotating.
    
    Arguments:
        `data_dict` (`dict[str, torch.Tensor]`):
            A dictionary containing the data to augment. The dictionary should have the following keys:
            - `data`: A tensor of shape `(B, T, H, W, C)`.
            - `params`: A tensor of shape `(B, ...)`.
    """
    data    = data_dict['data']
    assert data.ndim==5
    params  = data_dict['params']
    
    # Rotate data
    data    = torch.cat([torch.rot90(data, k, dims=(-3,-2)) for k in range(4)], dim=0)
    params  = torch.cat([params for _ in range(4)], dim=0)

    # Flip data
    data        = torch.cat([data, torch.flip(data, dims=(-2,))], dim=0)
    params      = torch.cat([params, params], dim=0)

    return {
        'data':     data,
        'params':   params,
    }


def augment_data_3D(data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Augment the data for 3D by flipping and rotating.
    
    Arguments:
        `data_dict` (`dict[str, torch.Tensor]`):
            A dictionary containing the data to augment. The dictionary should have the following keys:
            - `data`: A tensor of shape `(B, T, X, Y, Z, C)`.
            - `params`: A tensor of shape `(B, ...)`.
    """
    from    itertools       import  permutations
    dims    = (2, 3, 4)
    perms   = permutations(dims)
    data    = data_dict['data']
    assert data.ndim==6
    params  = data_dict['params']
    
    data = torch.cat([data.permute((0, 1, *p, 5)) for p in perms], dim=0)
    for d in dims:
        data = torch.cat([data, data.flip(d)], dim=0)
    params = torch.cat([params for _ in range(48)], dim=0)

    return {
        'data':     data,
        'params':   params,
    }
    

##################################################
##################################################
class WeightedDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            directory:  Union[str, Path],
            resolution: int,
            alpha:      Sequence[float],
            part_index: Sequence[int],
            dtype:      torch.dtype = torch.float,
            
            is_parameterized:       bool    = False,
            deg_param:              int     = 11,
            
            weighting_criterion:    Optional[float] = None,
            num_bins:               Optional[int]   = None,
        ) -> Self:
        # Check the input arguments
        if not (
            isinstance(weighting_criterion, Union[None, int, float]) and
            weighting_criterion >= 1
        ):
            raise ValueError(
                f"Invalid weighting criterion: {weighting_criterion}. It should be a positive real number not less than 1."
            )
        
        # Initialize the basic configurations
        super().__init__()
        self.__directory    = Path(directory) if not isinstance(directory, Path) else directory
        self.__resolution   = resolution
        self.__alpha        = alpha
        self.__part_index   = part_index
        self.__dtype_r      = dtype
        self.__is_parameterized     = is_parameterized
        self.__deg_param            = deg_param
        self.__weighting_criterion  = weighting_criterion
        self.__num_bins             = num_bins
        
        # Initialize auxiliary attributes
        self.__dimension:       Optional[int]   = None
        self.__v_max:           Optional[float] = None
        self.__v_where_closed:  Optional[str]   = None

        # Load the data
        self.__data:    Optional[torch.Tensor] = None
        self.__params:  Optional[torch.Tensor] = None
        
        # Precompute the weights
        self.__scores:  Optional[torch.Tensor] = None
        self.__density: Optional[torch.Tensor] = None
        self.__weights: Optional[torch.Tensor] = None
        self.__load_data()
        if not (weighting_criterion is None or num_bins is None):
            self.__precompute_scores()
            self.__precompute_weights()
        
        # Done
        return
    
    
    ##### Properties #####
    @property
    def dimension(self) -> Optional[int]:
        return self.__dimension
    @property
    def v_max(self) -> Optional[float]:
        return self.__v_max
    @property
    def v_where_closed(self) -> Optional[str]:
        return self.__v_where_closed
    
    @property
    def data(self) -> Optional[torch.Tensor]:
        return self.__data
    @property
    def params(self) -> Optional[torch.Tensor]:
        return self.__params
    @property
    def scores(self) -> Optional[torch.Tensor]:
        return self.__scores
    @property
    def density(self) -> Optional[torch.Tensor]:
        return self.__density
    @property
    def weights(self) -> Optional[torch.Tensor]:
        return self.__weights
    
    @property
    def num_bins(self) -> int:
        return self.__num_bins
    
    @property
    def transformed_params(self) -> Optional[torch.Tensor]:
        if not self.__is_parameterized:
            return None
        exp_speed = self.__params[:, [0]]
        r: torch.Tensor = torch.linspace(0, self.__v_max, self.__deg_param, dtype=self.__dtype_r)
        r = r.unsqueeze(0).repeat((exp_speed.size(0), 1))
        return torch.cat(
            [r, r*exp_speed, r.pow(exp_speed)],
            dim = 1,
        )
        
        
    def __load_data(self) -> None:
        from    itertools   import  product
        def _get_file_name(a: float, p: int) -> Path:
            res = str(self.__resolution).zfill(3)
            return self.__directory / f"res{res}__alpha{a:.1e}__part{p}.pth"
        
        file_list = [
            torch.load(_get_file_name(a, p), weights_only=False)
            for a, p in product(self.__alpha, self.__part_index)
        ]
        _list_of_v_max  = [f['v_max'] for f in file_list]
        _list_of_v_wc   = [f['v_where_closed'] for f in file_list]
        assert len(set(_list_of_v_max))==1, \
            "All files must have the same values of 'v_max'."
        assert len(set(_list_of_v_wc))==1, \
            "All files must have the same values of 'v_where_closed'."
        
        self.__data = \
            torch.cat([f['data'].type(self.__dtype_r) for f in file_list], dim=0)
        if self.__is_parameterized:
            self.__params = \
                torch.cat(
                    [
                        f['vhs_alpha'] * \
                        torch.ones((len(f['data']), 1), dtype=self.__dtype_r)
                        for f in file_list
                    ],
                    dim = 0,
                )
        
        self.__dimension        = (self.__data.ndim-3)//2
        self.__v_max            = float(_list_of_v_max[0])
        self.__v_where_closed   = str(_list_of_v_wc[0])
        
        return
    
    
    def __precompute_scores(self) -> None:
        # Prior configuration
        if self.__weighting_criterion is None:
            return
        from    deep_numerical.utils        import  velocity_grid, relative_error
        from    deep_numerical.numerical    import  distribution    as  D
        
        # Shape-related atributes
        old_shape = tuple(self.__data.shape)
        num_traj, num_steps = old_shape[:2]
        new_shape = tuple((num_traj*num_steps, *old_shape[2:]))
        
        # Compute the equilibrium distributions
        v_grid = velocity_grid(
            self.__dimension,
            self.__resolution,
            self.__v_max,
            where_closed    = self.__v_where_closed,
        )
        dist = self.__data.reshape(new_shape)
        equi = D.maxwellian_homogeneous(v_grid, *D.compute_moments_homogeneous(dist, v_grid))
        
        # Compute the scores based on the weighting criterion
        if isinstance(self.__weighting_criterion, Union[int, float]):
            """Compute the relative L^p error."""
            self.__scores = relative_error(dist, equi, p=self.__weighting_criterion)
        elif (
                isinstance(self.__weighting_criterion, str) and
                self.__weighting_criterion.lower()=='entropy'
            ):
            """Compute the entropy."""
            dv = 2*self.__v_max/self.__resolution
            ent_dist = D.compute_entropy_homogeneous(dist, dv)
            ent_equi = D.compute_entropy_homogeneous(equi, dv)
            self.__scores = ent_dist - ent_equi
        self.__scores = self.__scores.reshape((num_traj, num_steps)).log10()
                
        return
    
    
    def __precompute_weights(self) -> None:        
        # Classify the scores
        bins = torch.linspace(self.__scores.min(), self.__scores.max(), 1+self.__num_bins)
        bins[-1] = torch.inf; bins = bins[1:]  # Ensure the bins cover the entire range of scores.
        classes: torch.Tensor = torch.searchsorted(bins, self.__scores)
        unique_counts = torch.zeros((self.__num_bins,), dtype=torch.long)
        _unique_values, _unique_counts = classes.unique(return_counts=True)
        unique_counts[_unique_values] = _unique_counts
        
        # Compute the density and weights
        self.__density = unique_counts[classes]
        self.__density = self.__density / self.__density.sum()
        self.__weights = 1 / self.__density
        self.__weights = self.__weights / self.__weights.sum()
                
        return
    
    
    ##############################
    def __len__(self) -> int:
        return self.__data.size(0)
    
    def __getitem__(self, index: int) -> Sequence[torch.Tensor]:
        _idx_data = (index, slice(None), *(0 for _ in range(self.__dimension)))
        if self.__is_parameterized:
            if self.__is_parameterized:
                return (
                    self.__data[_idx_data],
                    self.transformed_params[index],
                    self.__weights[index],
                )
            else:
                return (
                    self.__data[_idx_data],
                    self.transformed_params[index],
                )
        else:
            if self.__is_parameterized:
                return (
                    self.__data[_idx_data],
                    self.__weights[index],
                )
            else:
                return (
                    self.__data[_idx_data],
                )
    ##############################
        
##################################################
##################################################
# End of file