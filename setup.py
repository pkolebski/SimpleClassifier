from distutils.core import setup

setup(name='SimpleClassifier',
      version='1.0',
      description='Simple Classifier',
      author='Piotr Kolebski',
      author_email='piotrkolebski@gmail.com',
      install_requires=[
          "click<=8.0.1",
          "Pillow<=8.3.2",
          "pytorch-lightning<=1.4.8"
          "scikit-learn<=0.24.2",
          "torch<=1.9.1",
          "torchvision<=0.10.1"
          "tqdm<=4.62.3"
      ],
      )
