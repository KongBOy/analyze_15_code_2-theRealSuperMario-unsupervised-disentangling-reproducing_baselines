import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import random
from edflow.iterators.batches import DatasetMixin
import skimage
import cv2
import os
from edflow.util import PRNGMixin
import PIL

cv2.setNumThreads(0)
# https://docs.chainer.org/en/stable/reference/generated/chainer.iterators.MultiprocessIterator.html


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        elem = l[i : i + n]
        random.shuffle(elem)
        yield elem


class Human3M_Dataset(Dataset):
    def __init__(self, path, chunk_size, n_shuffle):
        vids = [f for f in glob.glob(path + "*/*", recursive=True)]
        frames = []
        for vid in vids:
            for chunk in chunks(
                sorted(
                    glob.glob(vid + "/*.jpg", recursive=True),
                    key=lambda x: int(x.split("/")[-1].split(".jpg")[0]),
                ),
                chunk_size,
            ):
                if len(chunk) == chunk_size:
                    random.shuffle(chunk)
                    frames.append(chunk)

        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return self.frames[i]


class Toy_StochasticPairs(DatasetMixin):
    def __init__(self, config):
        super(Toy_StochasticPairs, self).__init__()

    def get_example(self, i):
        a = cv2.imread(
            "/export/home/sabraun/anaconda3/envs/computervisionbaselines/lib/python3.7/site-packages/skimage/data/astronaut.png",
            -1,
        )
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        # a = skimage.transform.resize(a, (128, 128))
        a = cv2.resize(a, (128, 128)) / 255.0
        b = a.copy()
        example = {"image_in": a, "image_rec": b}
        return example

    def __len__(self):
        return 1000


class Toy_Human3m(DatasetMixin):
    def __init__(self, config):
        super(Toy_Human3m, self).__init__()

        self.paths = (
            sorted(
                glob.glob(
                    "/export/home/sabraun/code/unsupervised-disentangling/datasets/toy_human3m/*.jpg"
                )
            )
            * 100
        )

    def get_example(self, i):
        p = self.paths[i]
        a = cv2.imread(p, -1,)
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        # a = skimage.transform.resize(a, (128, 128))
        a = cv2.resize(a, (128, 128)) / 255.0
        example = {"image_in": a, "image_rec": a}
        return example

    def __len__(self):
        return len(self.paths)


import pandas as pd


class StochasticPairs(DatasetMixin, PRNGMixin):
    def __init__(self, root, csv, spatial_size, id_colname, path_colname):
        self.spatial_size = spatial_size
        self.root = root
        self.csv = csv
        self.id_colname = id_colname
        self.path_colname = path_colname

        self.make_labels()
        self.labels = add_choices(self.labels, character_id_key=self.id_colname)
        self._length = len(self.labels[self.id_colname])

    def load_image(self, path):
        img = PIL.Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = np.asarray(img, dtype="float32")
        x = x / 255.0
        return x

    def make_labels(self):
        labels_df = pd.read_csv(self.csv)
        labels_df[self.path_colname] = labels_df[self.path_colname].apply(
            lambda x: os.path.join(self.root, x)
        )
        self.labels = dict(labels_df)
        self.labels = {k: list(v) for k, v in self.labels.items()}

    def __len__(self):
        return self._length

    def get_example(self, i):
        choices = self.labels["choices"][i]
        j = self.prng.choice(choices)

        view0 = self.load_image(self.labels[self.path_colname][i])
        view1 = self.load_image(self.labels[self.path_colname][j])

        view0 = cv2.resize(view0, (self.spatial_size, self.spatial_size))
        view1 = cv2.resize(view1, (self.spatial_size, self.spatial_size))

        batch = np.stack([view0, view1], axis=0)

        return {"image_in": batch}


class ExerciseTrain(StochasticPairs):
    def __init__(self, config):
        root = "/export/home/sabraun/code/unsupervised-disentangling/datasets/exercise_dataset"
        csv = "/export/home/sabraun/code/unsupervised-disentangling/datasets/exercise_dataset/csvs/instance_level_train_split.csv"
        spatial_size = config["spatial_size"]
        id_colname = "id"
        path_colname = "im1"
        super(ExerciseTrain, self).__init__(
            root, csv, spatial_size, id_colname, path_colname
        )


class ExerciseTest(StochasticPairs):
    def __init__(self, config):
        root = "/export/home/sabraun/code/unsupervised-disentangling/datasets/exercise_dataset"
        csv = "/export/home/sabraun/code/unsupervised-disentangling/datasets/exercise_dataset/csvs/instance_level_test_split.csv"
        spatial_size = config["spatial_size"]
        id_colname = "id"
        path_colname = "im1"
        super(ExerciseTest, self).__init__(
            root, csv, spatial_size, id_colname, path_colname
        )


class DeepfashionTrain(StochasticPairs):
    def __init__(self, config):
        root = "/export/home/sabraun/code/2020_parts/experiments/exp_03/secret_unsupervised_disentanglement/datasets/deepfashion_allJointsVisible/images"
        csv = "/export/home/sabraun/code/2020_parts/experiments/exp_03/secret_unsupervised_disentanglement/datasets/deepfashion_allJointsVisible/data_train.csv"
        spatial_size = config["spatial_size"]
        id_colname = "id"
        path_colname = "filename"
        super(DeepfashionTrain, self).__init__(
            root, csv, spatial_size, id_colname, path_colname
        )

    def get_example(self, i):
        choices = self.labels["choices"][i]

        view0 = self.load_image(self.labels[self.path_colname][i])
        view0 = np.pad(
            view0, ((20, 20), (20, 20), (0, 0)), constant_values=1.0
        )  # small padding to prevent head to get out of bound

        view0 = cv2.resize(view0, (self.spatial_size, self.spatial_size))

        batch = view0

        return {"image_in": batch}


# TODO: currently, there is no test split
class DeepfashionTest(StochasticPairs):
    def __init__(self, config):
        root = "/export/home/sabraun/code/2020_parts/experiments/exp_03/secret_unsupervised_disentanglement/datasets/deepfashion_allJointsVisible/images"
        csv = "/export/home/sabraun/code/2020_parts/experiments/exp_03/secret_unsupervised_disentanglement/datasets/deepfashion_allJointsVisible/data_test.csv"
        spatial_size = config["spatial_size"]
        id_colname = "id"
        path_colname = "filename"
        super(DeepfashionTest, self).__init__(
            root, csv, spatial_size, id_colname, path_colname
        )

    def get_example(self, i):
        choices = self.labels["choices"][i]

        view0 = self.load_image(self.labels[self.path_colname][i])
        view0 = np.pad(
            view0, ((20, 20), (20, 20), (0, 0)), constant_values=1.0
        )  # small padding to prevent head to get out of bound

        view0 = cv2.resize(view0, (self.spatial_size, self.spatial_size))

        batch = view0

        return {"image_in": batch}


def add_choices(labels, return_by_cid=False, character_id_key="character_id"):
    labels = dict(labels)
    cid_labels = np.asarray(labels[character_id_key])
    cids = np.unique(cid_labels)
    cid_indices = dict()
    for cid in cids:
        cid_indices[cid] = np.nonzero(cid_labels == cid)[0]
        verbose = False
        if verbose:
            if len(cid_indices[cid]) <= 1:
                print("No choice for {}: {}".format(cid, cid_indices[cid]))

    labels["choices"] = list()
    for i in range(len(labels[character_id_key])):
        cid = labels[character_id_key][i]
        choices = cid_indices[cid]
        labels["choices"].append(choices)
    if return_by_cid:
        return labels, cid_indices
    return labels
