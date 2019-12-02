import click
import os
from glob import glob

from skeleton.utils import build_skeleton_graph, load_skeleton, plot_skeleton_v2


def __load_and_plot(file_path, plot_dir):
    filename = os.path.basename(file_path).rsplit('.', 1)[0]
    print(filename)
    adjacency, coordinates = load_skeleton(file_path)
    graph = build_skeleton_graph(adjacency)
    plot_skeleton_v2(plot_dir + '/{}.png'.format(filename), graph, coordinates)


@click.command()
@click.option('--pkl_path',
              type=click.Path(exists=True),
              default='../data/train',
              show_default=True,
              help='pickle file path or directory of skeleton(s)')
@click.option('--plot_dir',
              type=click.Path(dir_okay=True),
              default='../data/visualization',
              show_default=True,
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
