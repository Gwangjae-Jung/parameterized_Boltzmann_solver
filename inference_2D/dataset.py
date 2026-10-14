from    typing                  import  TypeAlias, Literal
import  torch
from    torch.utils.data        import  TensorDataset
from    core.utils    import  GaussianNormalizer


NormKeys:   TypeAlias = Literal["data", "params"]


__all__: list[str] = ['nonparametric_dataset', 'parametric_dataset']


def nonparametric_dataset(
        data:   torch.Tensor,
        params: torch.Tensor,
        normalizers:    dict[NormKeys, GaussianNormalizer],
    ) -> TensorDataset:
    assert data.size(0)==params.size(0), \
        f"Data and parameters must have the same number of samples. Got {data.size(0)} samples for 'data' and {params.size(0)} samples for 'params'."
    
    dim = data.ndim-3   # Remove the batch, time, and channel dimensions
    res = data.size(-2)
    params = params.reshape(-1, *(1 for _ in range(dim+1)))
    params = params.repeat(1, *(res for _ in range(dim)), 1)
    
    dataset: dict[NormKeys, torch.Tensor] = {'data': data, 'params':params}
    for k in dataset.keys():
        dataset[k] = normalizers[k].encode(dataset[k])
    
    return TensorDataset(dataset['data'], dataset['params'])


def parametric_dataset(
        data:   torch.Tensor,
        params: torch.Tensor,
        normalizers:    dict[NormKeys, GaussianNormalizer],
    ) -> TensorDataset:
    assert data.size(0)==params.size(0), \
        f"Data and parameters must have the same number of samples. Got {data.size(0)} samples for 'data' and {params.size(0)} samples for 'params'."
    
    dataset: dict[NormKeys, torch.Tensor] = {'data': data, 'params':params}
    for k in dataset.keys():
        dataset[k] = normalizers[k].encode(dataset[k])
    
    return TensorDataset(dataset['data'], dataset['params'])