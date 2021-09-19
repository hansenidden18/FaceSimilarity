# Face imilarity Detection 

In this project i try to create a ***face verification*** program which recognize and cerify people faces, by using deepface library and face_recognition library. On the first program I used deep learning approach to verified the faces. And for the second program I used simple linear algebra equation which include on face_recognition library to verify the faces. The only issues that I facing was memory consumption on the deep learning program was quite big, so i need to clear my opened programs before runnning it. In the next step I will try to optimize my model and algorithm, so it can run in all device without lagging.

Live Demo for the Deep Learning approach is in this [link](https://youtu.be/u_zLsUfyZqI).


### [**FaceSimilarity**](https://github.com/hansenidden18/FaceSimilarity/blob/main/FaceSimilarity.py)

This program is a real time face similarity detection by using deepface function library and 3 models option. The models are VGG-Face, ArcFace, and Facenet. First, the model will extract your face using haar cascade algorithm which can be deleted if I don't want the face rectangle and it can also reude the program memory consumption. In the next step I will try to optimize my model and algorithm, so it can run in all device without lagging.


### [**FaceVerification**](https://github.com/hansenidden18/FaceSimilarity/blob/main/FaceVerification.py)

And this program is the perfection of my [**FaceSimilarity.py**](https://github.com/hansenidden18/FaceSimilarity/blob/main/FaceSimilarity.py) program which has some issues on memory consuming. In this program I use face_recognition library which so much lighter than my previous program. It can be happpened because the face recognition library already encode the known face of a person using dlib face recognition, and then compare the norm matrix of known face and the face from the camera.