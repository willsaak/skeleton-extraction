import click
import numpy as np
from pathlib import Path
from skimage import morphology
from tensorflow import keras


from nn.unet import unet
from skeleton.image_utils import load_image, show_image


@click.command()
@click.option('--inputs-path', type=click.Path(exists=True), default='../data/train', show_default=True)
@click.option('--targets-path', type=click.Path(exists=True), default='../data/visualization', show_default=True)
@click.option('--checkpoint', type=click.Path(exists=True, file_okay=True),
              default='../../models/2019-12-06-09-45-22/checkpoint.hdf5', show_default=True)
def main(inputs_path, targets_path, checkpoint):
    filepaths = sorted(Path(inputs_path).glob("*.png"))
    gt_folder = Path(targets_path)

    try:
        model = keras.models.load_model(checkpoint)
    except ValueError:
        model = unet(weights=checkpoint, batch_normalization=True)

    for filepath in filepaths:

        image = load_image(filepath)
        skeleton = model.predict(image[np.newaxis, ..., np.newaxis]).squeeze()

        static_skeleton = morphology.skeletonize((image > 0.5).astype(np.float32)).astype(np.float32)

        gt_filepath = gt_folder / filepath.name.replace('ip', 'gt')
        if gt_filepath.exists():
            gt = load_image(gt_filepath)
            show_image(gt)
        show_image(image)
        show_image(static_skeleton)
        show_image(skeleton)
        show_image((skeleton > 0.2).astype(np.float32))
        print()


if __name__ == '__main__':
    main()
