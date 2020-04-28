import pytest
import torch
from src.torch.utils import part_to_color_map, batch_colour_map


def test_part_to_color_map():
    # (2, 128, 128, 16)
    # (2, 64, 64, 16)
    # (2, 32, 32, 16)
    # (2, 16, 16, 80)
    # (2, 8, 8, 68)
    # (2, 4, 4, 66)
    n_channels = [16, 16, 16, 80, 68, 66]
    encoding_same_id = [
        torch.ones((2, 128 // (2 ** i), 128 // (2 ** i), c))
        for i, c in zip(range(len(n_channels)), n_channels)
    ]
    part_depths = [16, 16, 16, 16, 4, 2]
    in_dim = 128
    color_map_part = part_to_color_map(encoding_same_id, part_depths, size=in_dim)
    assert color_map_part.shape == (2, 3, 128, 128)


def test_batch_colour_map():
    part_maps = torch.ones((2, 16, 64, 64))
    out = batch_colour_map(part_maps)
    assert out.shape == (2, 3, 64, 64)
