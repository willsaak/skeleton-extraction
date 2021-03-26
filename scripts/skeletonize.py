import click
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
from skimage import morphology
from tensorflow import keras
from scipy import ndimage
from PIL import Image
from nn.unet import unet
from skeleton.image_utils import binarize_image, load_image, skeletonize_image, thin_image, plot_skeleton, show_image, \
    resize_image
from skeleton.extract_skeleton import extract_skeleton, extract_skeleton_v2, build_skeleton_graph
from scripts.predict import add_border, skeleton_union


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
    # checkpoint = '../../models/2020-01-13-11-05-02/checkpoint.hdf5'  # 512 with validation
    # checkpoint = '../../models/2020-02-01-02-08-28/checkpoint.hdf5'
    checkpoint = '../../models/2020-02-01-11-18-29/checkpoint.hdf5'
    # checkpoint_small = '../../models/2020-01-27-18-20-34/checkpoint.hdf5'  # 128 with validation
    # checkpoint_small = '../../models/2020-01-31-18-42-30/checkpoint.hdf5'  # 128 without validation
    checkpoint_small = '../../models/2020-01-31-19-29-01/checkpoint.hdf5'
    thickness = 2
    thickness_small = 1
    threshold = 0.09
    threshold_small = 0.4
    union_dist = 9

    image_folder, dst_folder = Path(image_folder), Path(dst_folder)

    input_images = sorted(image_folder.glob("*.png"))

    try:
        model = keras.models.load_model(checkpoint)
    except ValueError:
        model = unet(weights=checkpoint, batch_normalization=True)
    try:
        model_small = keras.models.load_model(checkpoint_small)
    except ValueError:
        model_small = unet(weights=checkpoint_small, batch_normalization=True)

    submission = {}
    for i, image_path in enumerate(input_images):
        # if int(str(image_path)[str(image_path).rfind('/') + 1:].split('_')[0]) not in [1034, 1052, 1066, 1004, 1053,
        #                                                                                1091, 1102, 1129, 1170, ]:
        #     continue
        print(f'Image {i} of {len(input_images)}...')
        id_ = image_path.stem.split('_')[0]

        image = load_image(image_path, size=(512, 512))
        image = add_border(image, thickness=thickness)
        # image = ndimage.gaussian_filter(image, sigma=1)
        # image = binarize_image(image, threshold=0.1)

        image_small = load_image(image_path, size=(128, 128))
        image_small = add_border(image_small, thickness=thickness_small)

        # static_skeleton = skeletonize_image(binarize_image(image, threshold=0.5))

        nn_skeleton = model.predict(image[np.newaxis, ..., np.newaxis]).squeeze()
        nn_skeleton = thin_image(binarize_image(nn_skeleton, threshold=threshold))

        nn_skeleton_small = model_small.predict(image_small[np.newaxis, ..., np.newaxis]).squeeze()
        # nn_skeleton_small = thin_image(binarize_image(nn_skeleton_small, threshold=0.07))

        # static_skeleton = skeleton_union(static_skeleton, nn_skeleton, pixel_count=parameter)
        # nn_skeleton = np.maximum(nn_skeleton, static_skeleton)
        nn_skeleton_small = resize_image(nn_skeleton_small, (512, 512))
        nn_skeleton_small = thin_image(binarize_image(nn_skeleton_small, threshold=threshold_small))
        static_skeleton = skeleton_union(nn_skeleton_small, nn_skeleton, pixel_count=union_dist)
        nn_skeleton = np.maximum(nn_skeleton, static_skeleton)
        # static_skeleton = skeleton_union(static_skeleton, nn_skeleton)
        # nn_skeleton = np.maximum(nn_skeleton, static_skeleton)
        # nn_adjacency, nn_coordinates = extract_skeleton_v2(nn_skeleton)
        skeleton = nn_skeleton
        # nn_adjacency, nn_coordinates = extract_skeleton_mst(nn_skeleton)

        # plot_skeleton(build_skeleton_graph(nn_adjacency), nn_coordinates, size=(512, 512)).show()

        # gt_adjacency, gt_coordinates = load_skeleton(ground_truth)
        # skeleton = skeletonize_image(binarize_image(image, threshold=0))
        # pred_adjacency, pred_coordinates = extract_skeleton(skeleton)

        # skeleton = model.predict(image[np.newaxis, ..., np.newaxis]).squeeze()
        # skeleton = thin_image(binarize_image(skeleton, threshold=0.07))
        pred_adjacency, pred_coordinates = extract_skeleton_v2(skeleton)

        # show_image(image)
        # show_image(skeleton)
        # plot_skeleton(build_skeleton_graph(pred_adjacency), pred_coordinates).show()

        submission[id_] = {'adjacency': pred_adjacency, 'coordinates': pred_coordinates.tolist()}
    dst_folder.mkdir(exist_ok=True, parents=True)
    dst_file = Path(dst_folder) / datetime.now().strftime('submission-%Y-%m-%d-%H-%M-%S.pkl')
    with dst_file.open('wb') as f:
        pickle.dump(submission, f)


if __name__ == '__main__':
    main()
