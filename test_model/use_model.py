import tensorflow as tf
import os
import keras
import numpy as np
import cv2
import sys
import argparse


from facial_recognition.Siamese_model import image_embedder, get_siamese_network




directory_path = os.getcwd()
database_path = os.path.join(directory_path, 'database')
encoder_path = os.path.join(directory_path, 'encoder.keras')

parser = argparse.ArgumentParser(description='gives filename')
parser.add_argument('file_name', metavar='file_name', type=str, help='gives filename')
args = parser.parse_args()


def read_image(Path):


    image = cv2.imread(Path, cv2.COLOR_BGR2RGB)
    image = tf.image.resize(image, (224,224))
    image = image / 255.0

    return image



encoder = image_embedder((224,224,3))

encoder.load_weights(encoder_path)
def classify_images(face_list1, face_list2, threshold=1.3):
    """

    :param face_list1:
    :param face_list2:
    :param threshold:
    :return:
    """
    # Getting the encodings for the passed faces
    tensor1 = encoder.predict(face_list1)
    tensor2 = encoder.predict(face_list2)

    distance = np.sum(np.square(tensor1 - tensor2), axis=-1)
    #prediction = np.where(distance <= threshold, 1, 0)
    if distance <= threshold:
        return 1
    else:
        return 0
file_name = args.file_name
test_path = os.path.join(directory_path, file_name)
img1 = np.array([read_image(Path=test_path)])
result = {}
for persons in os.listdir(database_path):
    result[persons] = []
    images_path = os.path.join(database_path, persons)
    for images in os.listdir(images_path):
        image_path = os.path.join(images_path, images)
        img_person = np.array([read_image(Path=image_path)])
        result1 = classify_images(img1, img_person)
        result[persons].append(result1)
print(result)
identified_person =0
max_match = 0
for persons in result.keys():
    match = np.sum(np.array(result[persons]) == 1)
    match_percentage = (match / len(result[persons]))*100
    if match_percentage >= max_match and match_percentage > 70:
        identified_person = persons
        max_match = match_percentage
    else:
        continue
if identified_person==0:
    unlock=False
    print("no match was found")
else:
    unlock=True
    print(f"identified person is {identified_person} with accuracy score {max_match}")






