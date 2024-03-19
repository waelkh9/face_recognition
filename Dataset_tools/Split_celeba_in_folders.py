import csv
import os
import shutil


def Read_Two_Column_File(file_path):
    """

    :param file_path: This is the file path
    :return: Two lists containing the labels and the filenames corresponding to each label.
    Y is the labels
    X is the filenames
    """
    with open(file_path, 'r') as f_input:
        csv_input = csv.reader(f_input, delimiter=' ', skipinitialspace=True)
        x = []
        y = []
        for cols in csv_input:
            x.append(cols[0])
            y.append(cols[1])

    return x, y

x, y = Read_Two_Column_File(r'/home/khlifi/Downloads/identity_CelebA.txt')

path = 'destination path'
src_folder = 'source folder path'
for i in range(len(x)):
    b=0
    for j in range(len(x)):


        if y[i] == y[j]:
            Path = os.path.join(path, y[j])
            src_path = os.path.join(src_folder, x[j])
            dest_path = os.path.join(Path, str(b)+".jpg")

            if not os.path.isdir(Path) and not os.path.isfile(dest_path) and os.path.isfile(src_path):
                print(f"Creating the new '{Path}' directory.")
                os.mkdir(Path)
                print(src_path)
                print(dest_path)


                shutil.copy(src_path, os.path.join(Path, str(b)+".jpg"))
                b=+1

            elif os.path.isdir(Path) and not os.path.isfile(dest_path) and os.path.isfile(src_path):
                shutil.copy(src_path, os.path.join(Path, str(b) + ".jpg"))


                b += 1
