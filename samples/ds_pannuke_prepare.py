import argparse
from dataclasses import dataclass
from pathlib import Path

from util.pannuke_datasets import process_fold

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Perform CellViT inference for given run-directory with model checkpoints and logs",
)
parser.add_argument(
    "--input_path",
    type=str,
    help="Input path of the original PanNuke dataset",
    required=True,
)
parser.add_argument(
    "--output_path",
    type=str,
    help="Output path to store the processed PanNuke dataset",
    required=True,
)

@dataclass
class Config:
    input_path: str
    output_path: str


args = Config(input_path='/Users/piotrwojcik/data/pannuke',
              output_path='/Users/piotrwojcik/data/pannuke')


if __name__ == "__main__":

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    for fold in [2]:
        process_fold(fold, input_path, output_path)