import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from skimage import morphology

from skeleton.utils import get_branches


def load_image(path: [Path, str], size=None) -> np.ndarray:
    image = Image.open(path).convert('L')
    if isinstance(size, tuple):
        image = image.resize(size)
    image = np.array(image, np.float32) / 255.
    return image


def resize_image(image: np.ndarray, size):
    nn_skeleton_small = Image.fromarray((image * 255).astype(np.uint8))
    nn_skeleton_small = nn_skeleton_small.resize(size)
    nn_skeleton_small = np.array(nn_skeleton_small, np.float32) / 255.
    return nn_skeleton_small


def show_image(image: np.ndarray):
    Image.fromarray((image * 255).astype(np.uint8)).show()


def plot_skeleton(graph, coordinates, size=(256, 256)) -> Image:
    image = Image.new('L', size)
    coordinates = np.stack([coordinates[:, 0] * image.size[1], (1 - coordinates[:, 1]) * image.size[0]], axis=-1)

    draw = ImageDraw.Draw(image)
    branches = get_branches(graph)
    for b in branches.values():
        draw.line([tuple(point.tolist()) for point in coordinates[b]], fill=255, width=1)

    return image


def binarize_image(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (image > threshold).astype(np.float32)


def thin_image(image: np.ndarray) -> np.ndarray:
    return morphology.thin(image).astype(np.float32)


def skeletonize_image(image: np.ndarray) -> np.ndarray:
    return morphology.skeletonize(image).astype(np.float32)
