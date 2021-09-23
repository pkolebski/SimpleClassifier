import csv

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from simple_classifier.data import DATA_PATH
from simple_classifier.dataset import SimpleDataset
from simple_classifier.trained_models import TRAINED_MODELS_PATH
from simple_classifier.training import Resnet34


def load_model(dataset: SimpleDataset, model_path: str):
    model = Resnet34(test_dataset=dataset)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


@click.command()
@click.option("--images_dir", default=str(DATA_PATH / "test_set"),
              help="Path to directory with images to predict", show_default=True)
@click.option("--output_file", default="output.csv", help="Path to file with prediction",
              show_default=True)
@click.option("--model_path", default=str(TRAINED_MODELS_PATH / "model.pth"),
              help="Path to trained model", show_default=True)
@click.option("--batch_size", default=10, help="Batch size", show_default=True)
def infer(images_dir: str, model_path: str, output_file: str, batch_size: int = 12):
    infer_transforms = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = SimpleDataset(images_dir, transforms=infer_transforms)
    data_loader = DataLoader(test_dataset, num_workers=1, batch_size=batch_size)
    model = load_model(test_dataset, model_path)
    eye = np.eye(3, dtype=int)
    with open(output_file, "w") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["file_name", "dog", "cat", "human"])
        for images, filenames in tqdm(data_loader):
            filenames = [f.split("/")[-1] for f in filenames]
            with torch.no_grad():
                preds = F.softmax(model(images))
                preds = np.argmax(preds.detach().numpy(), axis=1)
                for pred, f in zip(preds, filenames):
                    pred = [y for y in eye[pred]]
                    csv_writer.writerow([f, *pred])


if __name__ == "__main__":
    infer()