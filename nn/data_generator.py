import numpy as np
from tensorflow import keras
from typing import Callable, Collection, List, Optional, Tuple, Union


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


class DataGenerator(keras.utils.Sequence):
    x: np.ndarray
    y: np.ndarray
    class_mode: Optional[str]
    samples: int
    batch_size: int
    length: int
    shuffle: bool
    keep_remainder: bool

    _map_fn: Callable[[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]
    _batches: List[Tuple[int, int]]
    _indices: np.ndarray

    def __init__(self,
                 x: Union[Collection[np.ndarray], np.ndarray],
                 y: Union[Collection[np.ndarray], np.ndarray],
                 batch_size: int = 32,
                 shuffle: bool = True,
                 map_fn: Optional[Callable[[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]] = None,
                 keep_remainder: bool = True) -> None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"`batch_size` argument should be non-negative, but {batch_size} received.")
        self.batch_size = batch_size
        self.shuffle = bool(shuffle)
        self.keep_remainder = bool(keep_remainder)

        self.x = np.array(x)
        self.samples = len(self.x)
        if self.samples <= 0:
            raise ValueError(f"`x` argument should have at least one sample, but {self.samples} samples found.")

        self.y = np.array(y)
        if len(self.y) != self.samples:
            raise ValueError(f"Number of samples in `x` and `y` arguments should match, but {self.samples} and "
                             f"{len(self.y)} samples found.")
        print(f'Found {self.samples} samples.')

        if map_fn is None:
            self._map_fn = lambda batch: batch
        else:
            if not callable(map_fn):
                raise ValueError(f"`map_fn` argument should be callable.")
            self._map_fn = map_fn

        if self.keep_remainder:
            self.length = (self.samples + self.batch_size - 1) // self.batch_size
        else:
            self.length = self.samples // self.batch_size
            if self.length <= 0:
                raise ValueError(f"If `keep_remainder` argument is set to `False`, number of samples in `x` should be "
                                 f"greater or equal than `batch_size`.")

        self._batches = [(i * self.batch_size, min(self.samples, (i + 1) * self.batch_size))
                         for i in range(self.length)]
        self._indices = np.arange(self.samples)

        if self.shuffle:
            np.random.shuffle(self._indices)

    def __getitem__(self, index: int):
        batch_start, batch_end = self._batches[index]
        batch_indices = self._indices[batch_start:batch_end]
        batch = self._map_fn((self.x[batch_indices], self.y[batch_indices]))
        return batch

    def __len__(self):
        return self.length

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._indices)
