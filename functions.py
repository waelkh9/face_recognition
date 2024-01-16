import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import preprocess_input
import random

def get_imagee(image_path):

    #image_path=path = os.path.join(Path, str(image_path))
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    #image = tf.image.resize(image, (128,128))

    return image
def preprocess_images(anchor_img, positive_img, negative_img):
    return (preprocess_input(get_imagee(anchor_img)),preprocess_input(get_imagee(positive_img)),preprocess_input(get_imagee(negative_img)))


def split_dataset(directory, split=0.9):
    all_files = os.listdir(directory)
    len_train = int(len(all_files) * split)
    random.shuffle(all_files)
    train_list, test_list = {}, {}

    for folder in all_files[:len_train]:
        num_train = len(os.listdir(os.path.join(directory, folder)))
        train_list[folder] = num_train
    for folder in all_files[len_train:]:
        num_test = len(os.listdir(os.path.join(directory, folder)))
        test_list[folder] = num_test
    return train_list, test_list

def create_triplets(directory, folder_list, max_files=20):
    triplets = []
    list_folders = list(folder_list.keys())

    for folder in list_folders:
        path = os.path.join(directory, folder)
        files = list(os.listdir(path))[:max_files]
        num_files = len(files)

        for i in range(num_files - 1):
            for j in range(i + 1, num_files):
                anchor = (folder, f"{i}.jpg")
                positive = (folder, f"{j}.jpg")

                neg_folder = folder
                while neg_folder == folder:
                    neg_folder = random.choice(list_folders)
                neg_file = random.randint(0, folder_list[neg_folder] - 1)
                negative = (neg_folder, f"{neg_file}.jpg")

                triplets.append((anchor, positive, negative))

    random.shuffle(triplets)
    return triplets

def Generate_dataset(list, Path):
    anchor_label = []
    positive_label = []
    negative_label = []
    positive_image_path = []
    anchor_image_path = []
    negative_image_path = []
    for i in range(len(list)):

        a, p, n = list[i]
        anchor_path = os.path.join(Path, a[0], a[1])
        positive_path = os.path.join(Path, p[0], p[1])
        negative_path = os.path.join(Path, n[0], n[1])
        anchor_label.append(a[0])
        positive_label.append(p[0])
        negative_label.append(n[0])
        anchor_image_path.append(anchor_path)
        positive_image_path.append(positive_path)
        negative_image_path.append(negative_path)

    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_image_path)
    anchor_dataset = anchor_dataset.map(get_imagee)

    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_image_path)

    positive_dataset = positive_dataset.map(get_imagee)


    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_image_path)


    negative_dataset = negative_dataset.map(get_imagee)


    image_labels = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(anchor_label), tf.data.Dataset.from_tensor_slices(positive_label), tf.data.Dataset.from_tensor_slices(negative_label)))
    image_dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = tf.data.Dataset.zip((image_dataset, image_labels))
    return dataset
if __name__ == '__main__':

    print('function is running directly from original file')
else:
    print('functions is running from import')
