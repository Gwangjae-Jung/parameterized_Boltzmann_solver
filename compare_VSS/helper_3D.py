from    typing      import  Optional, Sequence, Union
from    pathlib     import  Path
import  yaml

import  torch
from    torch.utils.data    import  TensorDataset

from    deep_numerical.neuralop     import  \
    ModifiedFNO     as  FNO, \
    ModifiedRFNO    as  RFNO
from    deep_numerical.neuralop     import  PFNO, PRFNO
from    deep_numerical.utils        import  GaussianNormalizer

Models = Union[FNO, RFNO, PFNO, PRFNO]


##################################################
##################################################
# Keys of the models
K__FNO_PARAM    = "fno_param"
K__RFNO_PARAM   = "rfno_param"
K__PFNO_PARAM   = "pfno_param"
K__PRFNO_PARAM  = "prfno_param"
K__FNO_KERNEL   = "fno_kernel"
K__RFNO_KERNEL  = "rfno_kernel"
K__PFNO_KERNEL  = "pfno_kernel"
K__PRFNO_KERNEL = "prfno_kernel"

def get_model_acronym(model_key: str) -> str:
    """Return the acronym of the model in lowercases."""
    return model_key.split('_')[0].lower()
def get_model_parameterization(model_key: str) -> str:
    """Return the type of parameterization of the model of key `model_key`."""
    return {'param': "param_only", 'kernel': "with_kernel"}[model_key.split('_')[1].lower()]
def get_model_path(model_key: str) -> Path:
    """Return the base directory of the model."""
    base = Path(__file__).parent.parent
    model_name = get_model_acronym(model_key)
    basemodel_name = 'fno'
    param_type = get_model_parameterization(model_key)
    return base / f"{basemodel_name}__VSS__{param_type}" / f"{model_name.upper()}-Boltzmann_3D"


KEYS = tuple(
    [
        K__FNO_PARAM,
        K__RFNO_PARAM,
        K__PFNO_PARAM,
        K__PRFNO_PARAM,
        K__FNO_KERNEL,
        K__RFNO_KERNEL,
        K__PFNO_KERNEL,
        K__PRFNO_KERNEL,
    ]
)


CONFIG__PLOT_OF_LOSS: dict[str, dict[str, object]] = {
    K__FNO_PARAM:   {
        'color':        'black',
        'linestyle':    'dotted',
    },
    K__RFNO_PARAM:      {
        'color':        'red',
        'linestyle':    'dotted',
    },
    K__PFNO_PARAM:      {
        'color':        'blue',
        'linestyle':    'dotted',
    },
    K__PRFNO_PARAM:     {
        'color':        'orange',
        'linestyle':    'dotted',
    },
    K__FNO_KERNEL:      {
        'color':        'black',
        'linestyle':    'solid',
    },
    K__RFNO_KERNEL:     {
        'color':        'red',
        'linestyle':    'solid',
    },
    K__PFNO_KERNEL:     {
        'color':        'blue',
        'linestyle':    'solid',
    },
    K__PRFNO_KERNEL:    {
        'color':        'orange',
        'linestyle':    'solid',
    },
}


##################################################
##################################################
def initializer(model_name: str, model_cfg: dict) -> Models:
    """
    Returns a function that initializes the model with the given name.
    """
    if model_name in (K__FNO_PARAM, K__FNO_KERNEL):
        return FNO(**model_cfg)
    elif model_name in (K__RFNO_PARAM, K__RFNO_KERNEL):
        return RFNO(**model_cfg)
    elif model_name in (K__PFNO_PARAM, K__PFNO_KERNEL):
        return PFNO(**model_cfg)
    elif model_name in (K__PRFNO_PARAM, K__PRFNO_KERNEL):
        return PRFNO(**model_cfg)
    else:
        raise ValueError(f"Unknown model name: {model_name}")



def load_model_data(file_of_paths: Union[str, Path]) -> dict[str, dict]:
    # Load the paths of the configuration files
    with open(file_of_paths) as _p:
        info: dict = yaml.load(_p, Loader=yaml.FullLoader)

    model_data: dict[str, dict] = {}
    subkey_prefix   = 'prefix'
    subkey_datetime = 'datetime'
    subkey_seed     = 'seed'
    subkey_include  = 'include'
    
    # For each key, load the configuration
    for k in info.keys():
        # Pass the key if `include==False`
        if not info[k][subkey_include]:
            print(f"[{k}] The model is not included in the comparison.")
            continue
        if info[k][subkey_prefix] is None:
            _dt, _s = info[k][subkey_datetime], info[k][subkey_seed]
            info[k][subkey_prefix] = f"{_dt}__seed{_s}"
        prefix = info[k][subkey_prefix]

        model_name = get_model_acronym(k)
        model_path = get_model_path(k) / prefix
        _path_cfg:  Path = model_path / "config_train.yaml"
        _path_sd:   Path = model_path / f"{model_name}_boltzmann32_res32.pth"
        _path_norm: Path = model_path / f"{model_name}_boltzmann32_res32_normalizer.pth"
        _path_hist: Path = model_path / f"{model_name}_boltzmann32_res32.pickle"
        if not _path_cfg.exists():
            print(f"The configuration file for [{k}] is not loaded, as the given path does not exist.", f">>> {str(_path_cfg)}", "", sep='\n')
            continue
        if not _path_sd.exists():
            print(f"The model parameters for [{k}] are not loaded, as the given path does not exist.", f">>> {str(_path_sd)}", "", sep='\n')
            continue
        if not _path_norm.exists():
            print(f"The normalizers for [{k}] are not loaded, as the given path does not exist.", f">>> {str(_path_norm)}", "", sep='\n')
            continue
        if not _path_hist.exists():
            print(f"The history for [{k}] is not loaded, as the given path does not exist.", f">>> {str(_path_hist)}", "", sep='\n')
            continue
        
        # Initialize the storage, and load data
        model_data[k] = {}
        subkey_cfg  = 'config'
        subkey_sd   = 'state_dict'
        subkey_norm = 'normalizer'
        subkey_hist = 'history'
        with open(_path_cfg) as _cfg:
            model_data[k][subkey_cfg] = yaml.load(_cfg, Loader=yaml.FullLoader)
        model_data[k][subkey_sd]    = str(_path_sd)
        model_data[k][subkey_norm]  = str(_path_norm)
        model_data[k][subkey_hist]  = str(_path_hist)

    return model_data


def instantiate_models(
        model_data:     dict[str, dict],
        device:         Union[str, torch.device] = torch.device('cpu'),
    ) -> dict[str, Models]:
    """
    Instantiate models and move them to the specified device.
    """
    models: dict[str, Models] = {k: None for k in model_data.keys()}
    for k in models.keys():
        models[k] = initializer(k, model_data[k]['config']['model']).to(device)
        models[k].load_state_dict(torch.load(model_data[k]['state_dict'], weights_only=False))
    return models


def instantiate_normalizers(
        model_data:     dict[str, dict],
        device:         Union[str, torch.device] = torch.device('cpu'),
    ) -> dict[str, dict[str, GaussianNormalizer]]:
    """Instantiate normalizers and move them to the specified device.
    """
    normalizers: dict[str, dict[str, GaussianNormalizer]] = {k: {} for k in model_data.keys()}
    for k in normalizers.keys():
        normalizers[k] = torch.load(model_data[k]['normalizer'], weights_only=False)
        for v in normalizers[k].values():
            v.to(device)
    return normalizers


def instantiate_histories(model_data: dict[str, dict]) -> dict[str, dict[str, Sequence[float]]]:
    """Load the training histories from the specified paths in the model data.
    """
    import  pickle
    train_histories: dict[str, dict[str, Sequence[float]]] = {k: {} for k in model_data.keys()}
    for k in train_histories.keys():
        _train_history: dict[str, Union[list, float]]
        with open(model_data[k]['history'], 'rb') as k__hist:
            _train_history = pickle.load(k__hist)
        for j, w in _train_history.items():
            if isinstance(w, Union[list, tuple]):
                _train_history[j] = torch.Tensor(w)
        train_histories[k] = _train_history
    return train_histories


##################################################
##################################################
def construct_datasets(
        key:    str,
        
        data:           torch.Tensor,
        params:         torch.Tensor,
        normalizers:    Optional[dict[str, GaussianNormalizer]],
        
        v_max:          float,
        res_params:     Optional[int]   = None,
    ) -> dict[str, dict[str, TensorDataset]]:
    """Generates a dataset for the given model.
    """
    assert data.size(0)==params.size(0), \
        f"Data and parameters must have the same number of samples. Got {data.size(0)} samples for 'data' and {params.size(0)} samples for 'params'."
    
    model_name, parameterization_type = map(str.lower, key.split('_'))
    dataset: dict[str, torch.Tensor] = {'data': data, 'params': None}
    _dim = data.ndim-3
    _res = data.size(-2)
    
    if parameterization_type == 'param':
        if model_name[0] == 'p':
            # Parametric models without transform on the parameters
            from    train_utils.parameter_transformers  import  params_as_itselt
            dataset['params'] = params_as_itselt(params)
        else:
            # Non-parametric models without transform on the parameters
            from    train_utils.parameter_transformers  import  params_as_constant
            dataset['params'] = params_as_constant(params, _dim, _res)
    else:
        if model_name[0] == 'p':
            # Parametric models with transform on the parameters
            from    train_utils.parameter_transformers  import  params_as_combination
            dataset['params'] = params_as_combination(params, v_max, res_params)
        else:
            # Non-parametric models with transform on the parameters
            from    train_utils.parameter_transformers  import  params_as_power
            dataset['params'] = params_as_power(params, _dim, _res, v_max)
    
    if normalizers is not None:
        for k in dataset.keys():
            dataset[k] = normalizers[k].encode(dataset[k])
    
    return TensorDataset(dataset['data'], dataset['params'])




##################################################
##################################################
# End of file