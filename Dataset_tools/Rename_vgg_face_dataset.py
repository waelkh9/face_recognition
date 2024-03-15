import os
import shutil




#Path = "/home/khlifi/Documents/test_vgg"
Path = "/home/khlifi/Documents/test_bgg"
Folders = os.listdir(Path)
for folders in Folders:
    print(folders)
    new_path = os.path.join(Path, folders)
    list_of_files = os.listdir(new_path)
    #for i in range(10):
     #  print(list_of_files[i])
    for i in range(len(list_of_files)):
        old_path = os.path.join(Path, os.path.join(folders, list_of_files[i]))
        new_path = os.path.join(Path, os.path.join(folders, str(i)+".jpg"))
        os.rename(old_path, new_path)
