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


from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay


directory_path = os.getcwd()
Path = os.path.join(directory_path, 'Extracted Faces')
print(Path)
random.seed(5)
np.random.seed(5)
tf.random.set_seed(5)
folders = os.listdir(Path)

def read_imagee(index):
    """

    :param index: takes the  index of an image list .
    :return:a 3D tensor of the following shape will be returned [height, width, channels]
    """

    image_path = os.path.join(Path, index[0], index[1])
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (224,224))
def read_image(index):
    """

    :param index: takes the index to an image that contains the filename and path.
    :return: a 3D tensor of the following shape will be returned [height, width, channels]
    """

    path = os.path.join(Path, index[0], index[1])
    image = cv2.imread(path, cv2.COLOR_BGR2RGB)
    image = tf.image.resize(image, (224,224))
    image = image / 255.0

    return image




train_list, test_list = split_dataset(Path, 0.8)
print(f'this is the {test_list}')


test_triplet = create_triplets(Path, test_list, max_files=2)
print(f'this is the Triplet list  {test_triplet}')

def batch_generator(triplet_list, batch_size=256, preprocessing = False):
    """

    :param triplet_list: this is list containing the tripelets
    :param batch_size: size of the image batch
    :param preprocessing: specifies if any preprocessing is applied
    :return: returns a batch of images (images are in numpy array form).
    """
    batch_step = len(triplet_list)//batch_size

    for i in range(batch_step+1):
        anchor = []
        positive = []
        negative = []
        j = i*batch_size
        while j < (i+1)*batch_size and j < len(triplet_list):
            a, p, n = triplet_list[j]
            anchor.append(read_image(a)) # dont forget preprocess
            positive.append(read_image(p))
            negative.append(read_image(n))
            j+=1
        anchor = np.array(anchor, dtype="float")
        positive = np.array(positive, dtype="float")
        negative = np.array(negative, dtype="float")

        if preprocessing:
            anchor = preprocess_input(anchor)
            positive = preprocess_input(positive)
            negative = preprocess_input(negative)

        yield ([anchor, positive, negative])
def data_generator(triplet_list):
    """

    :param triplet_list: list of triplets in the following form (anchor, positive, negative)
    :return: returns a tuple containing the images that are in numpy array form.
    """
    batch_step = len(triplet_list)
    anchor = []
    positive = []
    negative = []
    for i in range(batch_step):

        a, p, n = triplet_list[i]
        anchor.append(read_image(a))
        positive.append(read_image(p))
        negative.append(read_image(n))
    anchor = np.array(anchor, dtype="float")
    positive = np.array(positive, dtype="float")
    negative = np.array(negative, dtype="float")



    yield ([anchor, positive, negative])


encoder = image_embedder((224,224,3))

encoder.load_weights(os.path.join('model', 'encoder.keras'))




def classify_images(face_list1, face_list2, threshold=1.2):
    """

    :param face_list1: This is a list of images can be anchor, positive or negative
    :param face_list2: this is a list of images, can be anchor , positive or negative
    :param threshold: A value for the euclidean distance that determines if two images
    belong to the same class.
    :return:
    """
    # Getting the encodings for the passed faces
    embeddings1 = encoder.predict(face_list1)
    embeddings2 = encoder.predict(face_list2)

    distance = np.sum(np.square(embeddings1 - embeddings2), axis=-1)
    prediction = np.where(distance <= threshold, 0, 1)
    return prediction


def ModelMetrics(pos_list, neg_list):
    """

    :param pos_list: Takes a list containing the predictions of the model on anchor and positive samples
    :param neg_list:Takes a list containing the predictions of the model on anchor and negative samples
    :return: a confusion matrix
    """
    true = np.array([0] * len(pos_list) + [1] * len(neg_list))
    pred = np.append(pos_list, neg_list)

    # Compute and print the accuracy
    print(f"\nAccuracy of model: {accuracy_score(true, pred)}\n")

    # Compute and plot the Confusion matrix

    cf_matrix = confusion_matrix(true, pred)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, )
    #disp.plot()
    #plt.show()
    categories = ['Similar', 'Different']
    names = ['True Similar', 'False Similar', 'False Different', 'True Different']
    percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
    plt.show()

pos_list = np.array([])
neg_list = np.array([])

for data in data_generator(test_triplet):

    a, p, n = data
    pos_list = np.append(pos_list, classify_images(a, p))

    neg_list = np.append(neg_list, classify_images(a, n))
    break

ModelMetrics(pos_list, neg_list)

