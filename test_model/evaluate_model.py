import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
import tensorflow as tf
import os
import random
import numpy as np
import cv2

from facial_recognition.Siamese_model import image_embedder
from facial_recognition.functions import split_dataset,create_triplets
import seaborn as sns
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import preprocess_input
from pathlib import Path

# Load the LFW pairs dataset
lfw_pairs = fetch_lfw_pairs(subset='test', color=True, resize=0.5)

# Prepare the images and labels
pairs = lfw_pairs.pairs
labels = lfw_pairs.target

# Preprocess images
def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    #image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image

# Load your pre-trained face recognition model
def load_model():
    # Replace this with the code to load your model
    model = tf.keras.models.load_model(r'C:\Users\waelk\PycharmProjects\facial_recognition\facial_recognition\test_model\model\encoder.keras')
    return model
encoder = image_embedder((224,224,3))

encoder.load_weights(os.path.join('model', 'encoder.keras'))
model = encoder

# Compute embeddings for pairs of images
def compute_embeddings(model, pairs):
    embeddings = []
    for i in range(pairs.shape[0]):
        img1 = preprocess_image(pairs[i, 0])
        img2 = preprocess_image(pairs[i, 1])
        embedding1 = model.predict(np.expand_dims(img1, axis=0))
        embedding2 = model.predict(np.expand_dims(img2, axis=0))
        embeddings.append((embedding1, embedding2))
    return embeddings

embeddings = compute_embeddings(model, pairs)

# Compute distance between embeddings and predict labels
def compute_distances(embeddings):
    distances = []
    for embedding1, embedding2 in embeddings:
        distance = np.linalg.norm(embedding1 - embedding2)
        distances.append(distance)
    return np.array(distances)

distances = compute_distances(embeddings)

# Threshold to classify pairs as same or different
threshold = 1.2  # You can tune this threshold based on a validation set

predicted_labels = distances < threshold
accuracy = accuracy_score(labels, predicted_labels)

print(f'Accuracy: {accuracy:.4f}')