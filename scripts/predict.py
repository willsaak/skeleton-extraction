import click
import numpy as np
from pathlib import Path
from tensorflow import keras


from postprocessing.extract_skeleton import load_image, show_image


@click.command()
@click.option('--src',
              type=click.Path(exists=True),
              default='../data/test',
              show_default=True)
@click.option('--checkpoint',
              type=click.Path(exists=True, file_okay=True),
              default='../../models/2019-12-02-14-31-24/checkpoint.hdf5',
              show_default=True)
def main(src, checkpoint):
    filepaths = sorted(Path(src).glob("*.png"))

    model = keras.models.load_model(checkpoint)
    # model.summary()

    for filepath in filepaths:
        image = load_image(filepath)
        skeleton = model.predict(image[np.newaxis, ..., np.newaxis]).squeeze()
        show_image(image)
        show_image(skeleton)
        print()


if __name__ == '__main__':
    main()
