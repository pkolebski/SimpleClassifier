# SimpleClassifier

## Instalation
```bash
pip install -e .
```

## Usage
1. Split the data set into a test, validation and training set.
```bash
python simple_classifier/utils/prepare_dataset.py
```
```bash
Usage: prepare_dataset.py [OPTIONS]

Options:
  --dataset_path TEXT  Path to downloaded nad extrcted dataset  [default: /hom
                       e/piotr/Projects/SimpleClassifier/simple_classifier/dat
                       a]
  --test_size FLOAT    Test set size in (0, 1>  [default: 0.2]
  --val_size FLOAT     Validation set size in (0, 1>  [default: 0.2]
  --help               Show this message and exit.

```
2. Run training:
```bash
python simple_classifier/training.py
```
```bash
Usage: training.py [OPTIONS]

Options:
  --train_dataset TEXT     Path to training dataset  [default: /home/piotr/Pro
                           jects/SimpleClassifier/simple_classifier/data/train
                           _set]
  --val_dataset TEXT       Path to validation dataset  [default: /home/piotr/P
                           rojects/SimpleClassifier/simple_classifier/data/val
                           _set]
  --test_dataset TEXT      Path to test dataset  [default: /home/piotr/Project
                           s/SimpleClassifier/simple_classifier/data/train_set
                           ]
  --annotations_path TEXT  Path to datasets annotations  [default: /home/piotr
                           /Projects/SimpleClassifier/simple_classifier/data/a
                           nnotations.json]
  --max_epoch INTEGER      Number of epochs  [default: 10]
  --batch_size INTEGER     Batch size  [default: 64]
  --gpus INTEGER           Number of GPUs to use  [default: 0]
  --help                   Show this message and exit.

```
Training result on test dataset:
```text
accuracy: 0.97,
f1 score: 0.97,
loss: 0.01
```

3. Run inference:
```bash
python simple_classifier/inference.py 
```
```bash
Usage: inference.py [OPTIONS]

Options:
  --images_dir TEXT     Path to directory with images to predict  [default: /h
                        ome/piotr/Projects/SimpleClassifier/simple_classifier/
                        data/test_set]
  --output_file TEXT    Path to file with prediction  [default: output.csv]
  --model_path TEXT     Path to trained model  [default: /home/piotr/Projects/
                        SimpleClassifier/simple_classifier/trained_models/mode
                        l.pth]
  --batch_size INTEGER  Batch size  [default: 10]
  --help                Show this message and exit.
```