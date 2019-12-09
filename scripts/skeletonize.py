import click
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
from skimage import morphology
from tensorflow import keras

from nn.unet import unet
from skeleton.image_utils import binarize_image, load_image, skeletonize_image, thin_image
from skeleton.extract_skeleton import extract_skeleton, extract_skeleton_v2


@click.command()
@click.option('--image-folder',
              type=click.Path(exists=True),
              default='../data/test',
              show_default=True)
@click.option('--dst-folder',
              type=click.Path(dir_okay=True),
              default='../data/submissions',
              show_default=True)
def main(image_folder, dst_folder):
    image_folder, dst_folder = Path(image_folder), Path(dst_folder)

    input_images = sorted(image_folder.glob("*.png"))

    checkpoint = '../../models/2019-12-06-09-45-22/checkpoint.hdf5'
    try:
        model = keras.models.load_model(checkpoint)
    except ValueError:
        model = unet(weights=checkpoint, batch_normalization=True)

    submission = {}
    for i, image_path in enumerate(input_images):
        print(f'Image {i} of {len(input_images)}...')
        id_ = image_path.stem.split('_')[0]

        image = load_image(image_path)

        # skeleton = skeletonize_image(binarize_image(image, threshold=0))
        # pred_adjacency, pred_coordinates = extract_skeleton(skeleton)

        skeleton = model.predict(image[np.newaxis, ..., np.newaxis]).squeeze()
        skeleton = thin_image(binarize_image(skeleton, threshold=0.15))
        pred_adjacency, pred_coordinates = extract_skeleton_v2(skeleton)
        # from skeleton.image_utils import plot_skeleton
        # from skeleton.utils import build_skeleton_graph
        # plot_skeleton(build_skeleton_graph(pred_adjacency), pred_coordinates).show()

        submission[id_] = {'adjacency': pred_adjacency, 'coordinates': pred_coordinates.tolist()}
    dst_folder.mkdir(exist_ok=True, parents=True)
    dst_file = Path(dst_folder) / datetime.now().strftime('submission-%Y-%m-%d-%H-%M-%S.pkl')
    with dst_file.open('wb') as f:
        pickle.dump(submission, f)


if __name__ == '__main__':
    main()
