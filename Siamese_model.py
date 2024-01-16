import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import backend, layers, metrics


from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, Sequential
from keras.layers import Dropout



def image_embedder(input_shape):
    """ Returns the convolutional neural network that will generate the encodings of each picture """

    pretrained_model = Xception(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )

    for i in range(len(pretrained_model.layers)-27):
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



def get_siamese_network(input_shape=(224, 224, 3)):
    encoder = image_embedder(input_shape)

    # Input Layers for the images
    anchor_input = layers.Input(input_shape, name="Anchor_Input")
    positive_input = layers.Input(input_shape, name="Positive_Input")
    negative_input = layers.Input(input_shape, name="Negative_Input")

    ## Generate the encodings (feature vectors) for the images)
    encoded_a = encoder(anchor_input)
    encoded_p = encoder(positive_input)
    encoded_n = encoder(negative_input)
    inputs = [anchor_input, positive_input, negative_input]
    outputs = [encoded_a, encoded_p, encoded_n]

    # Connect the inputs with the outputs
    siamese_triplet = tf.keras.Model(inputs=inputs, outputs=outputs)

    # return the model
    return encoder, siamese_triplet

if __name__ == '__main__':

    print('Siamese_model is running directly from original file')
else:
    print('Siamese_model is running from import')