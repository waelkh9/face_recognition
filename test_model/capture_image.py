import os

import cv2
from mtcnn import MTCNN
from facial_recognition.face_image_quality.face_image_quality import check_image_quality


directory_path = os.getcwd()
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")
img_counter= 0
while True:
    result, frame=cam.read()
    if not result:
        print("failed to access frame")
        break
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)
    if k%256==27:
        #if you press ESC
        print("ESC pressed, closing")
        break
    elif k%256==32:
        #if you press space
        img_name= f"test_image_new{img_counter}.jpg"
        img = cv2.imwrite(img_name, frame)
        image_path = directory_path + f"test_image_new{img_counter}.jpg"
        print(image_path)
        print(f"image {img_name}was saved")
        img_counter+= 1
cam.release()
cv2.destroyAllWindows()
print(img)
def crop_image(image_path):
    detector = MTCNN()
    img = cv2.imread(image_path)
    data = detector.detect_faces(img)
    biggest = 0
    if data != []:
        for faces in data:
            box = faces['box']
            # calculate the area in the image
            area = box[3] * box[2]
            if area > biggest:
                biggest = area
                bbox = box
        bbox[0] = 0 if bbox[0] < 0 else bbox[0]
        bbox[1] = 0 if bbox[1] < 0 else bbox[1]
        img = img[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert from bgr to rgb
        return (True, img)
    else:
        return (False, None)
img = crop_image(image_path)

x = check_image_quality(image_path)
if x==1 :
    print("image was accepted")
    cv2.imwrite("test_image_new.jpg", img)
else:
    print('image was not accepted')