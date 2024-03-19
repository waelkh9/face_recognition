from mtcnn import MTCNN
import cv2
import os
def crop_image(image_path):
    """

    :param image_path: Path to the input image
    :return: tuple containing a boolean indicating successful detection and the cropped
    image if successful otherwise (False, None)
    """
    detector = MTCNN()
    img=cv2.imread(image_path)
    data=detector.detect_faces(img)
    biggest=0
    if data !=[]:
        for faces in data:
            box=faces['box']
            # calculate the area of the bounding box in the image
            area = box[3] * box[2]
            #Checks if the current detected face is the largest
            if area>biggest:
                biggest=area
                bounding_box=box
        # Ensure bounding box coordinates are non-negative
        bounding_box[0]= 0 if bounding_box[0]<0 else bounding_box[0]
        bounding_box[1]= 0 if bounding_box[1]<0 else bounding_box[1]
        #crop  the face region from the imgae
        cropped_img =img[bounding_box[1]: bounding_box[1]+bounding_box[3],bounding_box[0]: bounding_box[0]+ bounding_box[2]]
        cropped_face = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB) # convert from bgr to rgb
        return (True, cropped_face)
    else:
        return (False, None)

Path2 = 'replace with source directory'
dst_path = 'replace with destination directory'
all_folders = os.listdir(Path2)
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