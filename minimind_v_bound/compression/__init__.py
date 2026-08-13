"""Quantization and description-length accounting."""

from .description_length import description_length_upper_bound
from .quantization import QuantizedVector, quantize_equal_width_bin_means

__all__ = [
    "QuantizedVector",
    "description_length_upper_bound",
    "quantize_equal_width_bin_means",
]

