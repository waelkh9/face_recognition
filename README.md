
# Access control with facial recognition

The aim of this work  is to create a an access control system using facial recognition. The model used in this project, was trained using triplet loss, specifically with online triplet mining.


## Running tests

The model can be found in the project directory called 'model'


The dataset is also included in the project directory path -> test_model\Extracted Faces

The directory test_model -> dataset contains the pictures of two individuals that will be used in testing. 

test_image_1 -> Person 1

test_image_2 -> Person 2

To use the model (saved in model), run this line:

```bash
py use_model.py 'filename'
```
for example : 
```bash
py use_model.py test_image_1.jpg
```

This will compare the input image 'test_image_1.jpg' to all images of the two persons in dataset directory. The result is expected to be Person 1.

To test the model use :
```bash
py test_model.py
```
## Depenedencies

```bash
  pip install requirements.txt
```

## RaspberryPI deployment

The script 'use_tflite.py' is the script that will run on the RaspberryPI, using the tensorflow lite model. It works in the same was as the use_model.py script.