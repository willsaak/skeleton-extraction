import click
import networkx as nx
import numpy as np
from pathlib import Path
from tensorflow import keras
from scipy import ndimage
from PIL import Image
from nn.unet import unet
from skeleton.extract_skeleton import extract_skeleton, extract_skeleton_v2, extract_skeleton_mst
from skeleton.image_utils import binarize_image, load_image, skeletonize_image, thin_image, plot_skeleton, show_image
from skeleton.utils import build_skeleton_graph, evaluate_skeleton, load_skeleton

import sys

sys.setrecursionlimit(3000)


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


def add_border(image: np.ndarray, thickness: int):
    image = image.copy()
    image[0:thickness, :] = 0.0
    image[:, 0:thickness] = 0.0
    image[-thickness:, :] = 0.0
    image[:, -thickness:] = 0.0
    return image


@click.command()
@click.option('--src', type=click.Path(exists=True), default='../data/train', show_default=True)
@click.option('--checkpoint', type=click.Path(exists=True, file_okay=True),
              default='../../models/2020-01-13-11-05-02/checkpoint.hdf5', show_default=True)
# default='../../models/2020-01-27-17-54-49/checkpoint.hdf5', show_default=True)
# default='../../models/2020-01-27-18-20-34/checkpoint.hdf5', show_default=True)
def main(src, checkpoint):
    import cProfile, pstats

    pr = cProfile.Profile()
    pr.enable()
    input_files = sorted(Path(src).glob("*.png"))
    ground_truths = sorted(Path(src).glob("*.pkl"))

    checkpoint = '../../models/2020-01-13-11-05-02/checkpoint.hdf5'  # 512 with validation
    # checkpoint_small = '../../models/2020-01-27-18-20-34/checkpoint.hdf5'  # 128 with validation
    # checkpoint_small = '../../models/2020-01-31-18-42-30/checkpoint.hdf5'  # 128 without validation
    checkpoint_small = '../../models/2020-01-31-19-29-01/checkpoint.hdf5'
    try:
        model = keras.models.load_model(checkpoint)
    except ValueError:
        model = unet(weights=checkpoint, batch_normalization=True)
    try:
        model_small = keras.models.load_model(checkpoint_small)
    except ValueError:
        model_small = unet(weights=checkpoint_small, batch_normalization=True)

    parameter2 = 4
    parameter = 15
    parameter3 = 11
    thickness = 2
    thickness_small = 1
    # for parameter2 in range(2, 6, 1):
    #     for parameter in range(7, 28, 2):
    #         for parameter3 in range(8, 15, 1):
    # for thickness_small in range(1, 2, 1):
    #     for thickness in range(2, 5, 1):
    nn_mean = 0
    nn_sum, static_sum = 0.0, 0.0
    for i, (input_file, ground_truth) in enumerate(zip(input_files, ground_truths)):
        input_file = Path('/home/william/Documents/Code/ml-comp-skeleton-sw/data/train/321_ip.png')
        ground_truth = Path('/home/william/Documents/Code/ml-comp-skeleton-sw/data/train/321_gt.pkl')
        image = load_image(input_file, size=(512, 512))
        # show_image(image)
        image = add_border(image, thickness=thickness)
        # show_image(image)

        image_small = load_image(input_file, size=(128, 128))
        # show_image(image_small)
        image_small = add_border(image_small, thickness=thickness_small)
        # show_image(image_small)
        # print()
        # image = ndimage.gaussian_filter(image, sigma=parameter2/2)
        # image = ndimage.gaussian_filter(image, sigma=0.5)
        # image = binarize_image(image, threshold=0.1)

        # static_skeleton = skeletonize_image(binarize_image(image, threshold=0.5))
        # static_adjacency, static_coordinates = extract_skeleton(static_skeleton)

        nn_skeleton = model.predict(image[np.newaxis, ..., np.newaxis]).squeeze()
        # show_image(nn_skeleton)
        nn_skeleton = thin_image(binarize_image(nn_skeleton, threshold=parameter3 / 100))
        # show_image(nn_skeleton)
        nn_skeleton_small = model_small.predict(image_small[np.newaxis, ..., np.newaxis]).squeeze()
        # nn_skeleton_small = thin_image(binarize_image(nn_skeleton_small, threshold=0.4))
        # show_image(nn_skeleton_small)
        # static_skeleton = skeleton_union(static_skeleton, nn_skeleton, pixel_count=parameter)
        # nn_skeleton = np.maximum(nn_skeleton, static_skeleton)
        nn_skeleton_small = Image.fromarray((nn_skeleton_small * 255).astype(np.uint8))
        nn_skeleton_small = nn_skeleton_small.resize((512, 512))
        nn_skeleton_small = np.array(nn_skeleton_small, np.float32) / 255.
        nn_skeleton_small = thin_image(binarize_image(nn_skeleton_small, threshold=parameter2 / 10))
        # show_image(nn_skeleton_small)
        static_skeleton = skeleton_union(nn_skeleton_small, nn_skeleton, pixel_count=parameter)
        nn_skeleton = np.maximum(nn_skeleton, static_skeleton)
        # show_image(nn_skeleton)

        nn_adjacency, nn_coordinates = extract_skeleton_v2(nn_skeleton)
        mst_adjacency, mst_coordinates = extract_skeleton_mst(nn_skeleton)

        # plot_skeleton(build_skeleton_graph(nn_adjacency), nn_coordinates, size=(512, 512)).show()

        gt_adjacency, gt_coordinates = load_skeleton(ground_truth)

        try:
            nn_score = evaluate_skeleton(build_skeleton_graph(gt_adjacency),
                                         build_skeleton_graph(nn_adjacency),
                                         gt_coordinates, nn_coordinates, num_samples=100, show_plot=True)
        except nx.exception.NetworkXPointlessConcept:
            nn_score = 0.0
        # try:
        #     static_score = evaluate_skeleton(build_skeleton_graph(gt_adjacency), build_skeleton_graph(static_adjacency),
        #                                      gt_coordinates, static_coordinates, num_samples=100, show_plot=False)
        # except nx.exception.NetworkXPointlessConcept:
        #     static_score = 0.0
        static_score = 83.97
        nn_sum += nn_score
        static_sum += static_score
        # print(f'Image: {input_file.name}, nn score: {nn_score:.02f}, morph score: {static_score:.02f}, '
        print(f'Image: {input_file}, nn score: {nn_score:.02f}, morph score: {static_score:.02f}, '
              f'mean nn score: {nn_sum / (i + 1):.02f}, mean morph score: {static_sum / (i + 1):.02f}')
        nn_mean = nn_sum / (i + 1)
    # print(f'thickness 1: {thickness} thickness small {thickness_small} nn_mean {nn_mean}')
    #     # print(f'Parameter value: {parameter}, parameter2 value: {parameter2}, parameter3 value: {parameter3} mean nn score: {nn_mean}')
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumulative')
    ps.print_stats()


if __name__ == '__main__':
    main()
