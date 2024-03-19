import tensorflow as tf
import os
import keras
from keras import backend, layers, metrics


from keras.applications import Xception
from keras.models import Model, Sequential
#os.chdir(r"C:\Users\waelk\PycharmProjects\facial_recognition\facial_recognition\test_model\model\encoder.keras")
def image_embedder(input_shape):
  """

  :param input_shape: take the input shape which the CNN will expect
  :return: The convolutional neural network that will be used in embedding.
  """
  """ Returns the convolutional neural network that will generate the encodings of each picture """

  pretrained_model = Xception(
    input_shape=input_shape,
    weights='imagenet',
    include_top=False,
    pooling='avg',
  )

  for i in range(len(pretrained_model.layers) - 27):
    pretrained_model.layers[i].trainable = False

  encode_model = Sequential([
    pretrained_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
  ], name="Encode_Model")
  return encode_model

model = image_embedder((224,224,3))
model.load_weights(r'C:\Users\waelk\PycharmProjects\facial_recognition\facial_recognition\test_model\model\encoder.keras')
saved_model_dir = r'C:\Users\waelk\PycharmProjects\facial_recognition\facial_recognition\test_model\model\encoder.keras'
# Convert the model
#encoder.summary()
#model.save('model_complete.keras')
#model = tf.keras.models.load_model(r'C:\Users\waelk\PycharmProjects\facial_recognition\facial_recognition\test_model\model_complete.keras', safe_mode=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]


tflite_model = converter.convert()

# Save the model.
with open('model/tf_lite_optimized_for_size.tflite', 'wb') as f:
  f.write(tflite_model)