import numpy as np
from pathlib import Path
from PIL import Image


def load_image(path: Path):
    image = np.array(Image.open(path).convert('L'))
    image = image.astype(np.float32) / 255.
    image = np.expand_dims(image, axis=-1)
    return image


def show_image(x: np.ndarray):
    Image.fromarray((x * 255).astype(np.uint8).squeeze()).show()


prediction_path = Path("/home/alexander/research/projects/ml-competition-2/visualization")
dst = Path("/home/alexander/research/projects/ml-competition-2/graphs")

preds = sorted([prediction_path / x for x in prediction_path.glob("*.png")])

for pred in preds:
    image = 1.0 - load_image(pred)
    show_image(image)
    print()