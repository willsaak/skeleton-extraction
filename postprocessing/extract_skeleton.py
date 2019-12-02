import click
import numpy as np
import pickle
from os import PathLike
from pathlib import Path
from PIL import Image
from skimage import morphology
from typing import List, Tuple

import skeleton.utils as skeleton_utils


def load_image(path: Path):
    image = np.array(Image.open(path).convert('L'))
    image = image.astype(np.float32) / 255.
    return image


def show_image(x: np.ndarray):
    Image.fromarray((x * 255).astype(np.uint8)).show()


def extract_skeleton_recursive(image: np.ndarray,
                               point: List[int],
                               prev_index: int,
                               coordinates: List[List[int]],
                               adjacency: List[List[int]]):
    x, y = point
    if -1 < x < image.shape[0] and -1 < y < image.shape[1] and image[x, y] == 1:
        try:
            coordinates.index(point)
        except ValueError:
            index = len(coordinates)
            coordinates.append(point)
            try:
                adjacency[prev_index].append(index)
                adjacency.append([prev_index])
            except IndexError:
                adjacency.append([])
            for next_point in [[x - 1, y - 1], [x - 1, y], [x - 1, y + 1], [x, y - 1],
                               [x, y + 1], [x + 1, y - 1], [x + 1, y], [x + 1, y + 1]]:
                extract_skeleton_recursive(image, next_point, index, coordinates, adjacency)
    return coordinates, adjacency


def choose_start_point(image: np.ndarray) -> List[int]:
    nodes = np.stack(np.nonzero(image), axis=-1)
    point = nodes[0]  # TODO: implement something meaningful.
    return point.tolist()


def extract_skeleton(image: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
    point = choose_start_point(image)

    coordinates, adjacency = extract_skeleton_recursive(image, point, -1, [], [])

    assert skeleton_utils.is_valid_skeleton(skeleton_utils.build_skeleton_graph(adjacency))

    # coordinates = np.array(coordinates, np.float32)

    coordinates = np.array(coordinates, np.float32) / image.shape

    # TODO: check and fix
    coordinates = np.flip(coordinates, axis=-1)
    coordinates[:, 1] = 1 - coordinates[:, 1]

    return adjacency, coordinates


def save_skeleton(path: PathLike, adjacency: List[List[int]], coordinates: np.ndarray):
    path = Path(path)
    assert not path.exists()
    skeleton = {'adjacency': adjacency, 'coordinates': coordinates.tolist()}
    with path.open('wb') as f:
        pickle.dump(skeleton, f)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    binary_image = (image > 0).astype(np.float32)
    thinned_image = morphology.thin(binary_image).astype(np.float32)
    # thinned_image = morphology.skeletonize(binary_image).astype(np.float32)
    return thinned_image


@click.command()
@click.option('--skeleton-image-path',
              type=click.Path(exists=True),
              default='/home/alexander/research/projects/ml-competition-2/visualization',
              show_default=True)
@click.option('--ground-truth-path',
              type=click.Path(exists=True),
              default='/home/alexander/research/projects/ml-competition-2/data/train',
              show_default=True)
def main(skeleton_image_path, ground_truth_path):
    skeleton_image_path, ground_truth_path = Path(skeleton_image_path), Path(ground_truth_path)

    skeleton_images = sorted([skeleton_image_path / filename for filename in skeleton_image_path.glob("*.png")])
    ground_truths = sorted([ground_truth_path / filename for filename in ground_truth_path.glob("*.pkl")])

    threshold = 0

    for image_path, gt_path in zip(skeleton_images, ground_truths):
        image = load_image(image_path)
        image = (image > threshold).astype(np.float32)
        image = morphology.thin(image).astype(np.float32)

        pred_adjacency, pred_coordinates = extract_skeleton(image)

        # nodes = np.nonzero(image)
        # a = np.zeros_like(image)
        # show_image(image)
        # a[nodes] = 1
        # show_image(a)

        # b = np.zeros_like(image)
        gt_adjacency, gt_coordinates = skeleton_utils.load_skeleton(gt_path)
        # gt_coordinates = np.round(gt_coordinates * image.shape).astype(np.int64)
        # b[255 - gt_coordinates[:, 1], gt_coordinates[:, 0]] = 1
        # show_image(b)
        # print()

        gt_graph = skeleton_utils.build_skeleton_graph(gt_adjacency)
        pred_graph = skeleton_utils.build_skeleton_graph(pred_adjacency)

        # skeleton_utils.plot_skeleton('1.png', gt_graph, gt_coordinates, show_branches=False)
        # skeleton_utils.plot_skeleton('2.png', pred_graph, pred_coordinates, show_branches=False)

        score = skeleton_utils.evaluate_skeleton(gt_graph, pred_graph, gt_coordinates, pred_coordinates,
                                                 num_samples=400, show_plot=True,
                                                 # plot_path=plot_dir + '/{}_{}.png'.format(gt_filename, pred_filename)
                                                 )
        # print(pred_filename, 'score:', score)
        print('score:', score)
        print()


if __name__ == '__main__':
    main()
