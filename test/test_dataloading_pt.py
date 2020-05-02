import pytest
from src.torch import dataloading_pt


def test_stochastic_pairs():
    root = (
        "/export/home/sabraun/code/unsupervised-disentangling/datasets/exercise_dataset"
    )
    csv = "/export/home/sabraun/code/unsupervised-disentangling/datasets/exercise_dataset/csvs/instance_level_train_split.csv"
    spatial_size = 128
    id_colname = "id"
    path_colname = "im1"
    dset = dataloading_pt.StochasticPairs(
        root, csv, spatial_size, id_colname, path_colname
    )

    ex = dset[0]

