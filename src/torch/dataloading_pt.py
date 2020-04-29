import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import random
from edflow.iterators.batches import DatasetMixin
import skimage
import cv2

# cv2.setNumThreads(0)
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
