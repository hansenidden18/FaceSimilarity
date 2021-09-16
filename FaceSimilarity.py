import numpy as np

import cv2

import time
from tqdm import tqdm

from matplotlib import pyplot as plt
from keras.models import load_model
import iproctor as mod
from deepface import DeepFace
from deepface.commons import functions

mod.CheckAll()

def build_model(model_name):

    global model_obj

    models = {
        'VGG-Face': mod.loadVGGModel,
        'ArcFace' : mod.loadArcModel,
        'Facenet' : mod.loadFacenetModel
    }

    if not "model_obj" in globals():
	    model_obj = {}

    if not model_name in model_obj.keys():
        model = models.get(model_name)
        if model:
            model = model()
            model_obj[model_name] = model
            #print(model_name," built")
        else:
            raise ValueError('Invalid model_name passed - {}'.format(model_name))

    return model_obj[model_name]

def CosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def euclidean_distance(source_representation, test_representation):   
    return np.sqrt(np.sum((np.array(source_representation) - np.array(test_representation))**2))

def represent(img_path, model_name = 'VGG-Face', model = None, enforce_detection = True, detector_backend = 'opencv', align = True, normalization = 'base'):

    if model is None:
        model = build_model(model_name)

    input_shape_x, input_shape_y = functions.find_input_shape(model)

    img = functions.preprocess_face(img = img_path
        , target_size=(input_shape_y, input_shape_x)
        , enforce_detection = enforce_detection
        , detector_backend = detector_backend
        , align = align)


    img = functions.normalize_input(img = img, normalization = normalization)

    
    embedding = model.predict(img)[0].tolist()

    return embedding

def similar(img1_path, img2_path = '', model_name = 'Face', distance_metric = 'cosine'):
    img1_representation = represent(img_path = img1_path
                        , model_name = model_name
                       
                        )
    img2_representation = represent(img_path = img2_path
                        , model_name = model_name
                
                        )
    distance = euclidean_distance(img1_representation,img2_representation)
    thresholds = {'VGG-Face': 0.6,
                'Facenet': 9,
                'ArcFace': 3.9}
    if distance <= thresholds[model_name]:
        return True
    else:
        return False
     


def extract_face(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors = 4)
    return face

def main():
    capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while True:
        ret, frame = capture.read()
        face = extract_face(frame)
        
        for (x1,y1,width,height) in face:
            cv2.rectangle(frame, (x1, y1), (x1+width , y1+height), (255, 0, 0), 2)

        distance = similar(frame,"Foto_Ketua_BDC_Hansen Idden.png",model_name='ArcFace')
        print(distance)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

main()