import click
import numpy as np
from pathlib import Path
from skimage import morphology
from tensorflow import keras
from scipy import ndimage
from PIL import Image
from nn.unet import unet
from skeleton.extract_skeleton import extract_skeleton, extract_skeleton_v2, extract_skeleton_mst
from skeleton.image_utils import binarize_image, load_image, skeletonize_image, thin_image, plot_skeleton, show_image, \
    resize_image
from skeleton.utils import build_skeleton_graph


def nearest_point(skeleton, x, y, pixel_count: int = 60):
    if np.sum(skeleton[max(x - pixel_count, 0):min(x + pixel_count, 512),
              max(0, y - pixel_count):min(512, y + pixel_count)]):
        return True
    return False


def skeleton_union(static_skeleton, nn_skeleton, pixel_count: int = 60):
    for x, row in enumerate(static_skeleton):
        for y, col in enumerate(row):
            if col and nearest_point(nn_skeleton, x, y, pixel_count):
                static_skeleton[x, y] = 0
    return static_skeleton


@click.command()
@click.option('--inputs-path', type=click.Path(exists=True), default='../data/train', show_default=True)
@click.option('--targets-path', type=click.Path(exists=True), default='../data/visualization', show_default=True)
@click.option('--checkpoint', type=click.Path(exists=True, file_okay=True),
              default='../../models/2020-01-13-11-05-02/checkpoint.hdf5', show_default=True)  # 512
# default='../../models/2020-01-27-17-54-49/checkpoint.hdf5', show_default=True) # 5
# default='../../models/2020-01-27-18-20-34/checkpoint.hdf5', show_default=True)
def main(inputs_path, targets_path, checkpoint):
    filepaths = sorted(Path(inputs_path).glob("*.png"))
    gt_folder = Path(targets_path)

    try:
        model = keras.models.load_model(checkpoint)
    except ValueError:
        model = unet(weights=checkpoint, batch_normalization=True)
    model_small = keras.models.load_model("../../models/2020-01-27-18-20-34/checkpoint.hdf5")
    for filepath in filepaths:
        filepath = Path(inputs_path) / '55_ip.png'
        image = load_image(filepath, size=(512, 512))
        image[0, :] = 0.0
        image[:, 0] = 0.0
        image[-1, :] = 0.0
        image[:, -1] = 0.0

        image_small = load_image(filepath, size=(128, 128))
        image_small[0, :] = 0.0
        image_small[:, 0] = 0.0
        image_small[-1, :] = 0.0
        image_small[:, -1] = 0.0
        # image = ndimage.gaussian_filter(image, sigma=parameter2/2)
        # image = ndimage.gaussian_filter(image, sigma=0.5)
        # image = binarize_image(image, threshold=0.1)

        # static_skeleton = skeletonize_image(binarize_image(image, threshold=0.5))
        # static_adjacency, static_coordinates = extract_skeleton(static_skeleton)

        nn_skeleton = model.predict(image[np.newaxis, ..., np.newaxis]).squeeze()
        nn_skeleton = thin_image(binarize_image(nn_skeleton, threshold=0.07))

        nn_skeleton_small = model_small.predict(image_small[np.newaxis, ..., np.newaxis]).squeeze()
        # nn_skeleton_small = thin_image(binarize_image(nn_skeleton_small1, threshold=0.07))
        show_image(nn_skeleton_small)
        # static_skeleton = skeleton_union(static_skeleton, nn_skeleton, pixel_count=parameter)
        # nn_skeleton = np.maximum(nn_skeleton, static_skeleton)
        nn_skeleton_small = resize_image(nn_skeleton_small, (512, 512))
        show_image(nn_skeleton_small)
        nn_skeleton_small = binarize_image(nn_skeleton_small, threshold=0.4)
        show_image(nn_skeleton_small)
        nn_skeleton_small = thin_image(nn_skeleton_small)
        show_image(nn_skeleton_small)
        static_skeleton = skeleton_union(nn_skeleton_small, nn_skeleton, pixel_count=7)
        nn_skeleton = np.maximum(nn_skeleton, static_skeleton)

        skeleton = model.predict(image[np.newaxis, ..., np.newaxis]).squeeze()

        # static_skeleton = morphology.skeletonize((image > 0.1).astype(np.float32)).astype(np.float32)
        # static_skeleton = skeleton_union(static_skeleton, thin_image(binarize_image(skeleton, threshold=0.07)))
        # union_skeleton = np.maximum(skeleton, static_skeleton)
        gt_filepath = gt_folder / filepath.name.replace('ip', 'gt')
        if gt_filepath.exists():
            gt = load_image(gt_filepath)
            show_image(gt)
        show_image(image)
        # show_image(static_skeleton)
        show_image(skeleton)
        show_image(nn_skeleton)
        # show_image(union_skeleton)
        # show_image((skeleton > 0.05).astype(np.float32))

        # nn_skeleton = thin_image(binarize_image(union_skeleton, threshold=0.07))
        # nn_adjacency, nn_coordinates = extract_skeleton_mst(nn_skeleton)
        # plot_skeleton(build_skeleton_graph(nn_adjacency), nn_coordinates, size=(512, 512)).show()
        print()


if __name__ == '__main__':
    main()
