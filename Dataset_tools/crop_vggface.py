from mtcnn import MTCNN
import cv2
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import imshow
import os
import shutil
def crop_image(image_path):
    detector = MTCNN()
    img=cv2.imread(image_path)
    data=detector.detect_faces(img)
    biggest=0
    if data !=[]:
        for faces in data:
            box=faces['box']
            # calculate the area in the image
            area = box[3] * box[2]
            if area>biggest:
                biggest=area
                bbox=box
        bbox[0]= 0 if bbox[0]<0 else bbox[0]
        bbox[1]= 0 if bbox[1]<0 else bbox[1]
        img =img[bbox[1]: bbox[1]+bbox[3],bbox[0]: bbox[0]+ bbox[2]]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert from bgr to rgb
        return (True, img)
    else:
        return (False, None)
#path = r"G:\datasets\celeb-A\img_align_celeba\000060.jpg"
Path2 = r"/home/khlifi/Documents/vgg_face/train"
dst_path = r"/home/khlifi/Documents/vggface_cropped"
all_folders = os.listdir(Path2)
print(len(all_folders))
result = False

for folders in all_folders:
    c=0
    folders_path = os.path.join(Path2, folders)
    files_list = os.listdir(folders_path)
    new_folder_path = os.path.join(dst_path, folders)
    if not os.path.exists(new_folder_path):
        os.mkdir(new_folder_path)
    for i in range(len(files_list)):

        img_path = os.path.join(folders_path, files_list[i])


        dst_path3 = os.path.join(new_folder_path,  str(c)+".jpg")
        if not os.path.isfile(dst_path3):
            result, images = crop_image(img_path)





            if result==True:
                images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
                cv2.imwrite(dst_path3, images)
                c += 1
            else:
                continue