#!pip install pytorch_lightning

from face_mobilenetv2 import FinalKeypointModel
import torch
import cv2
import numpy
import matplotlib.pyplot as plt
import time

hparams = {
    'batch_size' : 32,
    'learning_rate' : 0.0004
}  

loaded_model = FinalKeypointModel(hparams)
loaded_model.load_state_dict(torch.load('mobilenetv2_single_classif_layer.pt'))
loaded_model.eval()
print(loaded_model.device)

#loaded_model.cuda()

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#out = face_detect.detectMultiScale()

cap = cv2.VideoCapture(0)
frame_rate = 10
prev = 0

#ret, frame = cap.read()
#print(frame.shape)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")



while True:
    #ret, frame = cap.read()
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
    
    
    time_elapsed = time.time() - prev
    
    ret, frame = cap.read()
    if ret is False:
        cap.release()
        break
    '''
    SAVING A SINGLE IMAGE
    '''
    
    
    faces_rects = face_detect.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 11)
    
    if time_elapsed > 1./frame_rate:
        prev = time.time()

        faces_rects = face_detect.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 11)
        
        for (x,y,w,h) in faces_rects:
         #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
         #gray = cv2.cvtColor(frame[x- (w//10) : x+ 11*(w//10), y-(h//10) : y+ 11*(h//10), :],cv2.COLOR_BGR2GRAY)
         gray = cv2.cvtColor(frame[x: x+w, y : y+h, :],cv2.COLOR_BGR2GRAY)
         
         #HUE SHIFT
         
         #gray = gray + 20
         
         print()
         print(gray.dtype)
         print(gray.min())
         print(gray.max())
         print(f'({x},{y})')
         print(gray.shape)
         landmarks = loaded_model(gray).detach().numpy()
         print(landmarks.shape)
         
         #plt.figure()
         #plt.imshow(gray, cmap = 'gray',)
         #plt.scatter( list( 6*(w//10) + landmarks[0,:,0] * 6*(w//10)), list(6* (h//10) + landmarks[0,:,1] * 6*(h//10)), color='r')
         #plt.show()
         
         #w_ = w//10
         #h_ = h//10
         
         for row in range(landmarks[0].shape[0]):   #as the batch size was 1
             #cx = x + 6*w_ + round(float(landmarks[0][row][0] * 6 * w_))
             #cy = y + 6*h_ + round(float(landmarks[0][row][1] * 6 * h_))
             
             #cx = x + 5*(w//10) + int(round(landmarks[0][row][0] * 6*(w//10)))
             #cy = y + 5*(h//10) + int(round(landmarks[0][row][1] * 6*(h//10)))
             cx = int(round((landmarks[0][row][0]+1)/2 * w))
             cy = int(round((landmarks[0][row][1]+1)/2 * h))
             
             print(cx,cy)

             #cv2.circle(gray, 
             #           (cx , cy) ,
             #           0, 
             #           (0,0,255),
             #           4)
                
             cv2.circle(frame, 
                        (x + cx , y + cy) ,
                        0, 
                        (0,0,255),
                        4)
         #cv2.putText(frame, str('face'), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) 
         #cv2.rectangle(frame, (x- w//10, y-h//10), (x+ 11*(w//10), y+ 11*(h//10)), (0, 255, 0), 2)
         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
         
         cv2.imshow('Image', frame[x:x+w, y:y+h, :])
        #cv2.imshow('Image', frame)
        #cv2.imshow('Image',gray)
    c = cv2.waitKey(1)
    if c == 9:    #TAB to capture image
        plt.imsave('my_face.jpg',gray,cmap='gray')
    elif c == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
    
  