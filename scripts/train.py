import click
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras

from nn.data_generator import augment_batch, DataGenerator
from nn.focal_loss import balanced_binary_focal_loss
from nn.unet import unet
from skeleton.image_utils import load_image


@click.command()
@click.option('--inputs-path', type=click.Path(exists=True), default='../data/train', show_default=True)
@click.option('--targets-path', type=click.Path(exists=True), default='../data/visualization512', show_default=True)
@click.option('--dst', type=click.Path(), default='../../models', show_default=True)
@click.option('--epochs', type=int, default=200, show_default=True)
@click.option('--batch-size', type=int, default=2, show_default=True)
@click.option('--val-size', type=float, default=0.2, show_default=True)
@click.option('--checkpoint', type=click.Path(dir_okay=True), default='../../models/2020-02-01-09-20-56/checkpoint.hdf5')
def main(inputs_path, targets_path, dst, epochs, batch_size, val_size, checkpoint):
    input_files = sorted(Path(inputs_path).glob('*.png'))
    target_files = sorted(Path(targets_path).glob('*.png'))
    size = (512, 512)
    # size = (128, 128)
    # (train_input_files, val_input_files,
    #  train_target_files, val_target_files) = train_test_split(input_files, target_files, test_size=val_size)
    #
    # train_inputs = [load_image(x, size=size)[..., np.newaxis] for x in train_input_files]
    # train_targets = [load_image(x)[..., np.newaxis] for x in train_target_files]
    # val_inputs = [load_image(x, size=size)[..., np.newaxis] for x in val_input_files]
    # val_targets = [load_image(x)[..., np.newaxis] for x in val_target_files]

    train_inputs = [load_image(x, size=size)[..., np.newaxis] for x in input_files]
    train_targets = [load_image(x)[..., np.newaxis] for x in target_files]

    train_augmenter = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        fill_mode='constant',
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=None
    )

    train_generator = DataGenerator(train_inputs, train_targets, batch_size=batch_size, shuffle=True,
                                    map_fn=lambda x: augment_batch(x, train_augmenter)
                                    )
    # val_generator = DataGenerator(val_inputs, val_targets, batch_size=batch_size, shuffle=False)

    checkpoint = Path(checkpoint)
    if checkpoint.is_file():
        model = keras.models.load_model(checkpoint, compile=True)
    else:
        model = unet(batch_normalization=True, input_shape=(size[0], size[1], 1))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      # loss=balanced_binary_focal_loss(alpha=0.75, gamma=2.0),
                      metrics=['acc'])
    # model.summary()

    dst = Path(dst) / datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    dst.mkdir(exist_ok=False, parents=True)
    print(f"Save training to {dst}")

    checkpoint_callback = keras.callbacks.ModelCheckpoint(str(dst / 'checkpoint.hdf5'), monitor='loss', verbose=1,
                                                          save_best_only=True, save_weights_only=False)
    tensorboard_callback = keras.callbacks.TensorBoard(str(dst / 'summary'), write_graph=False)

    model.fit(train_generator, epochs=epochs,  # validation_data=val_generator,
              callbacks=[checkpoint_callback, tensorboard_callback],
              verbose=1)


if __name__ == '__main__':
    main()
