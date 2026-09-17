import pytest
import torch

from osd_masker.model import OSDMasker


@pytest.mark.parametrize("batch_size", [0, 1, 4])
@pytest.mark.parametrize("shape", [(64, 64), (192, 32), (8, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_model_input_output_compatible(batch_size: int, shape: tuple[int, int], dtype: torch.dtype) -> None:
    model = OSDMasker().to(dtype)
    full_shape = (batch_size, 1, *shape)
    image_in = torch.zeros(full_shape, dtype=dtype)
    image_out = model(image_in)
    assert image_out.dtype == dtype
    assert image_out.shape == full_shape


@pytest.mark.parametrize("shape", [(1, 16, 16), (1, 1, 1, 16, 16), (1, 1, 15, 16), (1, 1, 16, 15)])
def test_reject_incorrect_shapes(shape: tuple[int, ...]) -> None:
    model = OSDMasker()
    image_in = torch.zeros(shape)
    with pytest.raises(ValueError, match=r"Expected input with shape"):
        model(image_in)
