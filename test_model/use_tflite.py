import tensorflow as tf
import os
import keras
import numpy as np
import cv2
import sys
import argparse


#from Siamese_model import image_embedder, get_siamese_network
#from functions import split_dataset,get_image,Generate_dataset, create_triplets


from keras import layers
from keras.applications import Xception
from keras.models import Model, Sequential





directory_path = os.getcwd()
database_path = os.path.join(directory_path, 'database')

#parser = argparse.ArgumentParser(description='gives filename')
#parser.add_argument('file_name', metavar='file_name', type=str, help='gives filename')
#args = parser.parse_args()


def read_image(Path):
    """

    :param Path: Path to the image that will be read.
    :return: Returns a 3D tensor containing the image data in the following shape [widht,height,channels].
    """


    image = cv2.imread(Path, cv2.COLOR_BGR2RGB)
    image = tf.image.resize(image, (224,224))
    image = image / 255.0

    return image







def image_embedder(input_shape):
    """ Returns the image encoding model """

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

encoder = image_embedder((224,224,3))

#encoder.load_weights(r'C:\Users\waelk\PycharmProjects\facial_recognition\facial_recognition\test_model\model.tflite')

interpreter = tf.lite.Interpreter(model_path=r"C:\Users\waelk\PycharmProjects\facial_recognition\facial_recognition\test_model\model\tf_lite_model.tflite")
interpreter.allocate_tensors()



input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)
img1 = read_image(r'C:\Users\waelk\PycharmProjects\facial_recognition\facial_recognition\test_model\test_image_1.jpg')
tensor1 = interpreter.set_tensor(input_details[0]['index'], [img1])
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
img2 = read_image(r'C:\Users\waelk\PycharmProjects\facial_recognition\facial_recognition\test_model\test_image_2.jpg')
tensor2 = interpreter.set_tensor(input_details[0]['index'], [img2])
interpreter.invoke()
output_data2 = interpreter.get_tensor(output_details[0]['index'])
print(output_data2)
print(output_data)
def classify_images(face_list1, face_list2, threshold=1.2):
    """

    :param face_list1: a list of labeled imgaes
    :param face_list2:
    :param threshold:
    :return:
    """
    # Getting the encodings for the passed faces

    tensor1 = interpreter.set_tensor(input_details[0]['index'], [face_list1])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    tensor2 = interpreter.set_tensor(input_details[0]['index'], [face_list2])
    interpreter.invoke()
    output_data2 = interpreter.get_tensor(output_details[0]['index'])
    distance = np.sum(np.square(output_data - output_data2), axis=-1)
    #prediction = np.where(distance <= threshold, 1, 0)
    if distance <= threshold:
        return 1
    else: return 0
#file_name = args.file_name
test_path = os.path.join(directory_path, 'test_image_2.jpg')
img1 = read_image(Path=test_path)
result = {}
for persons in os.listdir(database_path):
    result[persons] = []
    images_path = os.path.join(database_path, persons)
    for images in os.listdir(images_path):
        image_path = os.path.join(images_path, images)
        img_person = read_image(Path=image_path)

        result1 = classify_images(img1, img_person)
        result[persons].append(result1)

identified_person = 0
max_match = 0
for persons in result.keys():
    match = np.sum(np.array(result[persons]) == 1)
    match_percentage = (match / len(result[persons]))*100
    if match_percentage >= max_match and match_percentage> 70:
        identified_person = persons
        max_match = match_percentage
    else:
        continue
if identified_person == 0:
    print("no matches were found")
else:
    print(f"identified person is {identified_person} with accuracy score {max_match}")






