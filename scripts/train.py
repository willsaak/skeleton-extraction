import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from typing import Tuple

from models.data_generator import DataGenerator
from models.unet import unet


def load_image(path: Path) -> np.ndarray:
    image = np.array(Image.open(path).convert('L'))
    image = image.astype(np.float32) / 255.
    image = np.expand_dims(image, axis=-1)
    return image


def augment_input_target(input_: np.ndarray,
                         target: np.ndarray,
                         augmenter: keras.preprocessing.image.ImageDataGenerator) -> Tuple[np.ndarray, np.ndarray]:
    transformation = augmenter.get_random_transform((256, 256, 1))

    augmented_input = augmenter.apply_transform(input_, transformation)
    augmented_target = augmenter.apply_transform(target, transformation)

    return augmented_input, augmented_target


def augment_batch(batch: Tuple[np.ndarray, np.ndarray],
                  augmenter: keras.preprocessing.image.ImageDataGenerator) -> Tuple[np.ndarray, np.ndarray]:
    augmented_batch_input = []
    augmented_batch_target = []
    for input_, target in zip(batch[0], batch[1]):
        i, t = augment_input_target(input_, target, augmenter)
        augmented_batch_input.append(i)
        augmented_batch_target.append(t)
    augmented_batch_input = np.stack(augmented_batch_input)
    augmented_batch_target = np.stack(augmented_batch_target)

    return augmented_batch_input, augmented_batch_target


input_path = Path("../data/train")
target_path = Path("../data/visualization")
dst = Path("../../models") / datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

val_size = 0.2
batch_size = 2
epochs = 100

input_data = sorted([x for x in input_path.glob("*.png")])
target_data = sorted([x for x in target_path.glob("*.png")])

input_data_train, input_data_val, target_data_train, target_data_test = train_test_split(input_data, target_data,
                                                                                         test_size=val_size)

input_data_train = np.stack([load_image(x) for x in input_data_train])
input_data_val = np.stack([load_image(x) for x in input_data_val])
target_data_train = np.stack([load_image(x) for x in target_data_train])
target_data_test = np.stack([load_image(x) for x in target_data_test])

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

train_generator = DataGenerator(input_data_train, target_data_train, batch_size=batch_size, shuffle=True,
                                map_fn=lambda x: augment_batch(x, train_augmenter))
val_generator = DataGenerator(input_data_val, target_data_test, batch_size=batch_size, shuffle=False)

checkpoint_callback = keras.callbacks.ModelCheckpoint(str(dst / 'checkpoint.hdf5'), monitor='val_loss', verbose=1,
                                                      save_best_only=True, save_weights_only=False)
tensorboard_callback = keras.callbacks.TensorBoard(str(dst / 'summary'), write_graph=False)

model = unet(batch_normalization=True)
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['acc'])

dst.mkdir(exist_ok=False, parents=True)
print(f"Save checkpoints to {dst}")

model.fit(train_generator, epochs=epochs, validation_data=val_generator,
          callbacks=[checkpoint_callback, tensorboard_callback], verbose=1)
