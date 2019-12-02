import click
import numpy as np
import os
from glob import glob
from PIL import Image, ImageDraw

from skeleton.utils import build_skeleton_graph, get_branches, load_skeleton
from postprocessing.extract_skeleton import load_image, show_image


def plot_skeleton(plot_path, graph, coordinates, show_branches=False):
    image = Image.new('L', (256, 256))
    coordinates = np.stack([coordinates[:, 0] * image.size[1], (1 - coordinates[:, 1]) * image.size[0]], axis=-1)

    draw = ImageDraw.Draw(image)
    branches = get_branches(graph)
    for b in branches.values():
        draw.line([tuple(point.tolist()) for point in coordinates[b]], fill=255, width=1)

    image.save(plot_path)


def __load_and_plot(file_path, plot_dir):
    filename = os.path.basename(file_path).rsplit('.', 1)[0]
    print(filename)
    adjacency, coordinates = load_skeleton(file_path)
    graph = build_skeleton_graph(adjacency)
    plot_skeleton(plot_dir + '/{}.png'.format(filename), graph, coordinates, show_branches=False)


@click.command()
@click.option('--pkl_path',
              default='/home/alexander/research/projects/ml-competition-2/data/train',
              help='pickle file path or directory of skeleton(s)')
@click.option('--plot_dir',
              default='/home/alexander/research/projects/ml-competition-2/visualization',
              help='plot directory')
def main(pkl_path, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    if os.path.isdir(pkl_path):
        for file_path in sorted(glob(pkl_path + '/*.pkl')):
            __load_and_plot(file_path, plot_dir)
    else:
        assert os.path.isfile(pkl_path)
        __load_and_plot(pkl_path, plot_dir)


if __name__ == '__main__':
    main()
