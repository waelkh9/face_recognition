from mtcnn import MTCNN
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
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
                bounding_box=box
        bounding_box[0]= 0 if bounding_box[0]<0 else bounding_box[0]
        bounding_box[1]= 0 if bounding_box[1]<0 else bounding_box[1]
        img =img[bounding_box[1]: bounding_box[1]+bounding_box[3],bounding_box[0]: bounding_box[0]+ bounding_box[2]]
        cropped_face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert from bgr to rgb
        return (True, cropped_face)
    else:
        return (False, None)
#path = r"G:\datasets\celeb-A\img_align_celeba\000060.jpg"
Path2 = r'/home/khlifi/Downloads/img_align_celeba'
dst_path= r"/home/khlifi/Documents/datasets"
all_files = os.listdir(Path2)
print(len(all_files))
for files in all_files:
    img_path = os.path.join(Path2, files)
    dst_path3 = os.path.join(dst_path, files)
    if not os.path.isfile(dst_path3):
        result, images = crop_image(img_path)
        print(files)
        print(img_path)
        if result==True:
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            cv2.imwrite(dst_path3, images)

