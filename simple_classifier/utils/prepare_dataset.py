import glob
import json
import os
import pathlib
from shutil import copy
from typing import Tuple, List, Union

import click
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from simple_classifier.data import DATA_PATH


def get_all_examples(data_path: pathlib.Path) -> Tuple[list[str], List[Tuple[int, int, int]]]:
    dogs_images = glob.glob(str(data_path / "train" / "dog" / "*"))
    dogs_annotations = [(1, 0, 0)] * len(dogs_images)
    cats_images = glob.glob(str(data_path / "train" / "cat" / "*"))
    cats_annotations = [(0, 1, 0)] * len(cats_images)
    humans_images = glob.glob(str(data_path / "train" / "human" / "*"))
    human_annotations = [(0, 0, 1)] * len(humans_images)
    all_images = dogs_images + cats_images + humans_images
    all_classes = dogs_annotations + cats_annotations + human_annotations
    return all_images, all_classes


def save_dataset(images: List[str], classes_: List[Tuple[int, int, int]], path: pathlib.Path):
    os.makedirs(path, exist_ok=True)
    annotations = dict()
    for image, annotation in tqdm(zip(images, classes_)):
        new_path = copy(image, path)
        annotations[new_path] = annotation
    return annotations


@click.command()
@click.option("--dataset_path", default=str(DATA_PATH),
              help="Path to downloaded nad extrcted dataset")
@click.option("--test_size", default=0.2, help="Test set size in (0, 1>")
@click.option("--val_size", default=0.2, help="Validation set size in (0, 1>")
def prepare_dataset(data_path: Union[pathlib.Path, str], test_size: float, val_size: float):
    if not isinstance(data_path, pathlib.Path):
        data_path = pathlib.Path(data_path)
    xs, ys = get_all_examples(data_path)
    train_x, other_x, train_y, other_y = train_test_split(
        xs,
        ys,
        test_size=test_size + val_size,
        random_state=42,
    )
    val_x, test_x, val_y, test_y = train_test_split(
        other_x,
        other_y,
        test_size=test_size / (test_size + val_size),
        random_state=42,
    )

    annotations = save_dataset(train_x, train_y, data_path / "train_set")
    annotations.update(save_dataset(test_x, test_y, data_path / "test_set"))
    annotations.update(save_dataset(val_x, val_y, data_path / "val_set"))
    with open(data_path / "annotations.json", "w") as file:
        json.dump(annotations, file, indent=4)


if __name__ == "__main__":
    prepare_dataset()
