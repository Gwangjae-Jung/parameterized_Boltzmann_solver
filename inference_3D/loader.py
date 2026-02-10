import  torch
from    pathlib     import  Path
from    typing      import  Any, TypeAlias, Literal, Union
from    deep_numerical.neural.parameterized_op  import  ModifiedFNO, ModifiedRFNO, PFNO, PRFNO


__all__: list[str] = ['ModelKeys', 'load_model_data', 'path_of_state_dict']


##################################################
ModelKeys:  TypeAlias = Literal["FNO2D-param", "RFNO2D-param", "PFNO2D-param", "PRFNO2D-param"]
DataKeys:   TypeAlias = Literal["state_dict", "normalizer", "train_history"]
NormKeys:   TypeAlias = Literal["data", "params"]
HistKeys:   TypeAlias = Literal["train_rel_error", "train_rel_error_cons", "val_rel_error", "val_rel_error_cons", "train_time"]
ModelTypes: TypeAlias = Union[ModifiedFNO, ModifiedRFNO, PFNO, PRFNO]
path_base = Path(r"/media/junseung/47a90e46-3a9d-467c-bbee-066752b68532/GWANGJAE/parameterized_Boltzmann_solver")


##################################################
#####< Model parameters >#####
# `in_channels` should be specified for each model
base_config: dict[str, Any] = {
    'hidden_channels':    32,
    'out_channels':       1,
    
    'lift_layer':         [128, 128],
    'n_layers':           4,
    'project_layer':      [128, 128],
    
    'activation_name':    "relu",
    'activation_kwargs':  {},
    
    'pos_enc':            True,
    
    # 'delta_t':            1.0,      # The default config
}


##################################################
def load_model_args(
        model_key:      ModelKeys,
        max_freq:       int,
        dim_domain:     int = 3,
        dim_parameters: int = 1,
    ) -> dict:
    """Returns the arguments for initializing the specified model."""
    config = base_config.copy()
    
    # Parametric or not
    in_channels: int
    if model_key in ["FNO3D-param", "RFNO3D-param"]:
        in_channels = 2
    elif model_key in ["PFNO3D-param", "PRFNO3D-param"]:
        in_channels = 1
        config['dim_parameters'] = dim_parameters
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    # Radial or not
    if model_key=="RFNO3D-param" or model_key=="PRFNO3D-param":
        config['dim_domain'] = dim_domain
        config['max_freq'] = max_freq
    else:
        config['n_modes'] = [2*max_freq+1 for _ in range(dim_domain)]
        
    config['in_channels'] = in_channels
    return config


def path_of_model(model_key: ModelKeys, max_freq: int, seed: int) -> Path:
    """Returns the *path* of the model-related data. This data contains the trained parameters, normalizer, and training history."""
    return path_base / "fno__VHS__param_only" / model_key /  f"{model_key}__max_freq{max_freq:02d}__seed{seed}"
    

def load_model_data(model_key: ModelKeys, max_freq: int, seed: int) -> dict[DataKeys, Any]:
    """Returns the model-related data. This data contains the trained parameters, normalizer, and training history."""
    _path = path_of_model(model_key, max_freq, seed)
    try:
        ret = {
            "state_dict":   torch.load(_path / "state_dict.pth", weights_only=False),
            "normalizer":   torch.load(_path / "normalizer.pth", weights_only=False),
            "train_history":torch.load(_path / "train_history.pth", weights_only=False),
        }
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The path {_path.parts[-4:]} cannot be found.")
    return ret