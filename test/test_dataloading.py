from src.tf import dataloading
from dotmap import DotMap


def test_load_train_human3m():
    arg = DotMap({"chunk_size": 2, "n_shuffle": 1})

    path = "../datasets/human3/train"
    raw_dataset = dataloading.load_train_human3m(arg, path)


def test_load_test_human3m():
    arg = DotMap({"chunk_size": 2, "n_shuffle": 1})

    path = "../datasets/human3/train"
    raw_dataset = dataloading.load_test_human3m(arg, path)


def test_load_train_generic():
    arg = DotMap({"chunk_size": 2, "n_shuffle": 1})

    path = "../datasets/human3/train"
    raw_dataset = dataloading.load_train_generic(arg, path)


def test_load_test_generic():
    arg = DotMap({"chunk_size": 2, "n_shuffle": 1})

    path = "../datasets/human3/train"
    raw_dataset = dataloading.load_test_generic(arg, path)


def test_load_csv():
    arg = DotMap({"chunk_size": 2, "n_shuffle": 1})
    raw_dataset = dataloading.load_train_from_csv(
        arg, ".", "toy_dataset.csv", "id", "im1"
    )
    iterator = raw_dataset.make_one_shot_iterator()
    elem = iterator.get_next()

