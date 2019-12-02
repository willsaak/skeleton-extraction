import click
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from typing import Optional, Tuple

from models.data_generator import DataGenerator
from models.unet import unet


def load_image(path: Path) -> np.ndarray:
    image = np.array(Image.open(path).convert('L'))
    image = image.astype(np.float32) / 255.
    image = np.expand_dims(image, axis=-1)
    return image


def augment_batch(
        batch: Tuple[np.ndarray, np.ndarray],
        augmenter: keras.preprocessing.image.ImageDataGenerator,
        seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    inputs, targets = batch

    img_shape = inputs.shape[1:]
    augmented_inputs, augmented_targets = [], []
    for input_, target in zip(inputs, targets):
        transform = augmenter.get_random_transform(img_shape, seed=seed)

        augmented_input = augmenter.apply_transform(input_, transform)
        augmented_target = augmenter.apply_transform(target, transform)

        augmented_inputs.append(augmented_input)
        augmented_targets.append(augmented_target)
    augmented_inputs = np.stack(augmented_inputs)
    augmented_targets = np.stack(augmented_targets)

    return augmented_inputs, augmented_targets


# input_path = Path("../data/train")
# target_path = Path("../data/visualization")
# dst = Path("../../models") / datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#
# val_size = 0.2
# batch_size = 1
# epochs = 2

@click.command()
@click.option('--inputs-path',
              type=click.Path(exists=True, dir_okay=True),
              default='../data/train',
              show_default=True)
@click.option('--targets-path',
              type=click.Path(exists=True, dir_okay=True),
              default='../data/visualization',
              show_default=True)
@click.option('--dst',
              type=click.Path(dir_okay=True),
              default='../../models',
              show_default=True)
@click.option('--epochs',
              type=int,
              default=2,
              show_default=True)
@click.option('--batch-size',
              type=int,
              default=1,
              show_default=True)
@click.option('--val-size',
              type=float,
              default=0.2,
              show_default=True)
def main(inputs_path, targets_path, dst, epochs, batch_size, val_size):
    input_data = sorted(Path(inputs_path).glob("*.png"))
    target_data = sorted(Path(targets_path).glob("*.png"))

    input_data_train, input_data_val, target_data_train, target_data_test = train_test_split(input_data, target_data,
                                                                                             test_size=val_size)

    inputs_train = [load_image(x) for x in input_data_train]
    inputs_val = [load_image(x) for x in input_data_val]
    targets_train = [load_image(x) for x in target_data_train]
    targets_val = [load_image(x) for x in target_data_test]

    train_augmenter = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=None,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode='constant',
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=None
    )

    train_generator = DataGenerator(inputs_train, targets_train, batch_size=batch_size, shuffle=True,
                                    map_fn=lambda x: augment_batch(x, train_augmenter))
    val_generator = DataGenerator(inputs_val, targets_val, batch_size=batch_size, shuffle=False)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(str(dst / 'checkpoint.hdf5'), monitor='val_loss', verbose=1,
                                                          save_best_only=True, save_weights_only=False)
    tensorboard_callback = keras.callbacks.TensorBoard(str(dst / 'summary'), write_graph=False)

    model = unet(batch_normalization=True)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    dst = Path(dst) / datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    dst.mkdir(exist_ok=False, parents=True)
    print(f"Save training to {dst}")

    model.fit(train_generator, epochs=epochs, validation_data=val_generator,
              callbacks=[checkpoint_callback, tensorboard_callback], verbose=1)


if __name__ == '__main__':
    main()
