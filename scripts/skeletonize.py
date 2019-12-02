import click
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
from skimage import morphology

from postprocessing.extract_skeleton import load_image, extract_skeleton


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

    input_images = sorted([filename for filename in image_folder.glob("*.png")])
    # ground_truths = sorted([filename for filename in image_folder.glob("*.pkl")])

    threshold = 0

    submission = {}
    for i, image_path in enumerate(input_images):
        print(f'Image {i} of {len(input_images)}...')
        id_ = image_path.stem.split('_')[0]

        image = load_image(image_path)
        image = (image > threshold).astype(np.float32)

        # TODO: replace `morphology.skeletonize` with NN-prediction, then postprocess the prediction
        image = morphology.skeletonize(image).astype(np.float32)
        # image = model.predict(np.expand_dims(image, axis=-1)).squeeze(axis=-1)
        # image = (image > threshold).astype(np.float32)
        # morphology.thin(binary_image).astype(np.float32)

        pred_adjacency, pred_coordinates = extract_skeleton(image)

        # gt_adjacency, gt_coordinates = skeleton_utils.load_skeleton(ground_truths[i])
        # gt_graph = skeleton_utils.build_skeleton_graph(gt_adjacency)
        # pred_graph = skeleton_utils.build_skeleton_graph(pred_adjacency)
        #
        # # skeleton_utils.plot_skeleton('1.png', gt_graph, gt_coordinates, show_branches=False)
        # # skeleton_utils.plot_skeleton('2.png', pred_graph, pred_coordinates, show_branches=False)
        #
        # score = skeleton_utils.evaluate_skeleton(gt_graph, pred_graph, gt_coordinates, pred_coordinates,
        #                                          num_samples=400, show_plot=True,
        #                                          # plot_path=plot_dir + '/{}_{}.png'.format(gt_filename, pred_filename)
        #                                          )
        # # print(pred_filename, 'score:', score)
        # print('score:', score)
        # print()

        submission[id_] = {'adjacency': pred_adjacency, 'coordinates': pred_coordinates.tolist()}
    dst_folder.mkdir(exist_ok=True, parents=True)
    dst_file = Path(dst_folder) / datetime.now().strftime('submission-%Y-%m-%d-%H-%M-%S.pkl')
    with dst_file.open('wb') as f:
        pickle.dump(submission, f)


if __name__ == '__main__':
    main()
