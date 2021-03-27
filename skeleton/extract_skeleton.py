import pickle
from os import PathLike
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from skeleton.utils import build_skeleton_graph, is_valid_skeleton


def arg_nearest_coordinates(x: np.ndarray, y: np.ndarray) -> Tuple[int, int]:
    """
    Find indices of the nearest coordinates in two given arrays.

    Args:
        points1, array of shape (len_x, dim): First array of coordinates.
        points2, array of shape (len_y, dim): Second array of coordinates.

    Return:
        Tuple of two indices, -1 if at least one of the input arrays is empty.
    """
    if len(x) > 0 and len(y) > 0:
        distances = cdist(x, y, 'euclidean')
        index_x, index_y = np.unravel_index(np.argmin(distances), distances.shape)
        return index_x, index_y
    return -1, -1


def choose_start_point(image: np.ndarray,
                       prev_coordinates: np.ndarray = np.empty((0, 2), np.int)) -> Tuple[int, int]:
    """
    Choose starting point of a skeleton.
    """
    coordinates = np.stack(np.nonzero(image), axis=-1)
    _, index = arg_nearest_coordinates(prev_coordinates, coordinates)
    if index < 0 and len(coordinates) > 0:
        # index = 0  # Try something meaningful?
        center = np.mean(coordinates, axis=0, keepdims=True)
        _, index = arg_nearest_coordinates(center, coordinates)
    point = tuple(coordinates[index].tolist())
    return point


def extract_skeleton_recursive(image: np.ndarray,
                               point: Tuple[int, int],
                               prev_index: int,
                               adjacency: List[List[int]],
                               coordinates: List[Tuple[int, int]]) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    y, x = point
    if -1 < y < image.shape[0] and -1 < x < image.shape[1] and image[y, x] == 1:
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
            for next_point in ((y - 1, x - 1), (y - 1, x), (y - 1, x + 1), (y, x - 1),
                               (y, x + 1), (y + 1, x - 1), (y + 1, x), (y + 1, x + 1)):
                adjacency, coordinates = extract_skeleton_recursive(image, next_point, index, adjacency, coordinates)
    return adjacency, coordinates


def extract_skeleton(image: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
    """
    Extract a single connected skeleton from binary image.
    """
    point = choose_start_point(image)

    adjacency, coordinates = extract_skeleton_recursive(image, point, -1, [], [])

    if len(coordinates) > 1:
        assert is_valid_skeleton(build_skeleton_graph(adjacency))

    if len(coordinates) == 1:
        adjacency = []
    coordinates = np.array(coordinates, np.float) / image.shape
    coordinates = np.stack([coordinates[:, 1], 1 - coordinates[:, 0]], axis=-1)

    return adjacency, coordinates


def merge_skeletons(adjacency: List[List[int]],
                    coordinates: np.ndarray,
                    new_adjacency: List[List[int]],
                    new_coordinates: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
    """
    Merge two skeletons.
    """
    index, index_new = arg_nearest_coordinates(coordinates, new_coordinates)
    if index >= 0 and index_new >= 0:
        adjacency[index].append(len(adjacency) + index_new)
        for i in range(len(new_adjacency)):
            for j in range(len(new_adjacency[i])):
                new_adjacency[i][j] += len(adjacency)
        new_adjacency[index_new].append(index)
    adjacency += new_adjacency
    coordinates = np.concatenate([coordinates, new_coordinates])
    return adjacency, coordinates


def extract_skeleton_v2(image: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
    """
    Extract all probably not connected skeletons from binary image and merge them through the nearest points.
    """
    image = image.copy()
    coordinates, adjacency = np.empty((0, 2), np.int), []
    while True:
        try:
            point = choose_start_point(image, coordinates)
        except IndexError:
            break

        new_adjacency, new_coordinates = extract_skeleton_recursive(image, point, -1, [], [])
        new_coordinates = np.array(new_coordinates, np.int)
        image[tuple(new_coordinates.T)] = 0

        if len(new_coordinates) > 1:
            assert is_valid_skeleton(build_skeleton_graph(new_adjacency))

        adjacency, coordinates = merge_skeletons(adjacency, coordinates, new_adjacency, new_coordinates)

        if len(new_coordinates) > 1:
            assert is_valid_skeleton(build_skeleton_graph(adjacency))

    if len(coordinates) == 1:
        adjacency = []
    coordinates = coordinates.astype(np.float) / image.shape
    coordinates = np.stack([coordinates[:, 1], 1 - coordinates[:, 0]], axis=-1)

    return adjacency, coordinates


def save_skeleton(path: PathLike, adjacency: List[List[int]], coordinates: np.ndarray):
    path = Path(path)
    assert not path.exists()
    skeleton = {'adjacency': adjacency, 'coordinates': coordinates.tolist()}
    with path.open('wb') as file:
        pickle.dump(skeleton, file)


def extract_skeleton_mst(image: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
    coordinates = np.stack(np.nonzero(image), axis=-1)
    distances = cdist(coordinates, coordinates, 'euclidean')
    from scipy.sparse.csgraph import minimum_spanning_tree
    tcsr = minimum_spanning_tree(distances)
    tcsr = tcsr.toarray().astype(int)
    tcsr += tcsr.T
    adjacency = [np.nonzero(col)[0].tolist() for col in tcsr]

    if len(coordinates) > 1:
        assert is_valid_skeleton(build_skeleton_graph(adjacency))

    if len(coordinates) == 1:
        adjacency = []
    coordinates = np.array(coordinates, np.float) / image.shape
    coordinates = np.stack([coordinates[:, 1], 1 - coordinates[:, 0]], axis=-1)

    return adjacency, coordinates
