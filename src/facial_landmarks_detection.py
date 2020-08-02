
import os
import cv2
from openvino.inference_engine import IECore
from util_function import  preprocess_input

class Model_facial_landmarks_detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device):
        '''
        Set instance variables.
        '''
        self.device=device
        self.model=model_name
        self.model_structure=model_name
        self.model_weights=os.path.splitext(self.model_structure)[0] + ".bin"
        
        try:
            self.model=IECore().read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        
    def load_model(self):
        '''
        Load the model to the specified device.
        '''     

        core = IECore()
        
        self.Network = core.load_network(network=self.model, device_name=self.device)
        

    def predict(self, image, face, face_coords, display):
        '''
        Run predictions on the input image.
        '''

        p_frame = preprocess_input(face, self.input_shape)
        
        self.Network.start_async(request_id=0, inputs={self.input_name: p_frame})
        
        if self.Network.requests[0].wait(-1) == 0:
            
            outputs = self.Network.requests[0].outputs[self.output_name]
        
            image, left_eye, right_eye, eyes_center = self.preprocess_output(outputs, face_coords, image, display)
        
        return image, left_eye, right_eye, eyes_center       
        
    def preprocess_output(self, outputs, face_coords, image, display):
        '''
        Preprocess the output before feeding it to the next model.
        '''

        landmarks = outputs.reshape(1, 10)[0]

        height = face_coords[3] - face_coords[1]
        width = face_coords[2] - face_coords[0]
        
        x_l = int(landmarks[0] * width) 
        y_l = int(landmarks[1]  *  height)
        
        xmin_l = face_coords[0] + x_l - 30
        ymin_l = face_coords[1] + y_l - 30
        xmax_l = face_coords[0] + x_l + 30
        ymax_l = face_coords[1] + y_l + 30
         
        x_r = int(landmarks[2]  *  width)
        y_r = int(landmarks[3]  *  height)
        
        xmin_r = face_coords[0] + x_r - 30
        ymin_r = face_coords[1] + y_r - 30
        xmax_r = face_coords[0] + x_r + 30
        ymax_r = face_coords[1] + y_r + 30
        
        if(display):
 
            cv2.rectangle(image, (xmin_l, ymin_l), (xmax_l, ymax_l), (0,0,255), 2)        
            cv2.rectangle(image, (xmin_r, ymin_r), (xmax_r, ymax_r), (0,0,255), 2)
        

        left_eye_center =[face_coords[0] + x_l, face_coords[1] + y_l]
        right_eye_center = [face_coords[0] + x_r , face_coords[1] + y_r]      
        eyes_center = [left_eye_center, right_eye_center ]
        
        left_eye = image[ymin_l:ymax_l, xmin_l:xmax_l]
        
        right_eye = image[ymin_r:ymax_r, xmin_r:xmax_r]
        
        return image, left_eye, right_eye, eyes_center