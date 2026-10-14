"""Neural network layers for core.
"""
from core.neural.layer.general              import MLP, HyperMLP, PatchEmbedding, Periodization1D
from core.neural.layer.fourier_layer        import SpectralConv, FourierLayer
from core.neural.layer.fourier_layer_radial import RadialSpectralConv, RadialFourierLayer

__all__: list[str] = [
    "MLP",
    "HyperMLP",
    "PatchEmbedding",
    "Periodization1D",
    "SpectralConv",
    "FourierLayer",
    "RadialSpectralConv",
    "RadialFourierLayer",
]


def main() -> None:
    pass


if __name__ == '__main__':
    main()
