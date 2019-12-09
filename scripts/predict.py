import click
import networkx as nx
import numpy as np
from pathlib import Path
from tensorflow import keras

from nn.unet import unet
from skeleton.extract_skeleton import extract_skeleton, extract_skeleton_v2
from skeleton.image_utils import binarize_image, load_image, skeletonize_image, thin_image
from skeleton.utils import build_skeleton_graph, evaluate_skeleton, load_skeleton


@click.command()
@click.option('--src', type=click.Path(exists=True), default='../data/train', show_default=True)
@click.option('--checkpoint', type=click.Path(exists=True, file_okay=True),
              default='../../models/2019-12-06-09-45-22/checkpoint.hdf5', show_default=True)
def main(src, checkpoint):
    input_files = sorted(Path(src).glob("*.png"))
    ground_truths = sorted(Path(src).glob("*.pkl"))

    try:
        model = keras.models.load_model(checkpoint)
    except ValueError:
        model = unet(weights=checkpoint, batch_normalization=True)

    nn_sum, static_sum = 0.0, 0.0

    for i, (input_file, ground_truth) in enumerate(zip(input_files, ground_truths)):
        image = load_image(input_file)

        static_skeleton = skeletonize_image(binarize_image(image, threshold=0.5))
        static_adjacency, static_coordinates = extract_skeleton(static_skeleton)

        nn_skeleton = model.predict(image[np.newaxis, ..., np.newaxis]).squeeze()
        nn_skeleton = thin_image(binarize_image(nn_skeleton, threshold=0.15))
        nn_adjacency, nn_coordinates = extract_skeleton_v2(nn_skeleton)

        gt_adjacency, gt_coordinates = load_skeleton(ground_truth)

        try:
            nn_score = evaluate_skeleton(build_skeleton_graph(gt_adjacency), build_skeleton_graph(nn_adjacency),
                                         gt_coordinates, nn_coordinates, num_samples=100, show_plot=False)
        except nx.exception.NetworkXPointlessConcept:
            nn_score = 0.0
        try:
            static_score = evaluate_skeleton(build_skeleton_graph(gt_adjacency), build_skeleton_graph(static_adjacency),
                                             gt_coordinates, static_coordinates, num_samples=100, show_plot=False)
        except nx.exception.NetworkXPointlessConcept:
            static_score = 0.0
        nn_sum += nn_score
        static_sum += static_score
        print(f'Image: {input_file.name}, nn score: {nn_score:.02f}, morph score: {static_score:.02f}, '
              f'mean nn score: {nn_sum / (i + 1):.02f}, mean morph score: {static_sum / (i + 1):.02f}')


if __name__ == '__main__':
    main()
