from tensorflow import keras
import numpy as np
from PIL import Image

model_path = "/home/william/Documents/Code/models/2019-12-04-17-08-37/checkpoint.hdf5"
middle = keras.models.load_model(model_path, custom_objects={"backend": keras.backend})
file_path = "/home/william/Documents/Code/ml-comp-skeleton-extraction/data/train/225_ip.png"#"/home/william/Documents/Code/ml-comp-skeleton-extraction/data/train/628_ip.png"
image = np.array(Image.open(file_path).convert('L'))
image = image.astype(np.float32) / 255.
image = np.expand_dims(image, axis=-1)
x = middle.predict(np.expand_dims(image, axis=0))
Image.fromarray((x * 255).astype(np.uint8).squeeze()).show()
embedding1 = middle.predict(np.expand_dims(image / 255., axis=0))[0][0]
embedding2 = middle.predict(np.expand_dims(np.fliplr(image) / 255., axis=0))[0][0]
