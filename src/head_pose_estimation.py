
import os
import cv2
from math import cos, sin, pi
from openvino.inference_engine import IECore
from util_function import  preprocess_input

class Model_head_pose_estimation:
    '''
    Class for the Head Pose Estimation  Model.
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
        self.output_name= [i for i in self.model.outputs.keys()]
               
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
            
            outputs = self.Network.requests[0].outputs
            
            out_image,  head_angles  = self.preprocess_output(image, outputs, face_coords, display)
            
        return out_image,  head_angles 
        
    def draw_outputs(self, image, head_angle ,face_coords): 
        '''
        Draw model output on the image.
        '''
        
        cos_r = cos(head_angle[2] * pi / 180)
        sin_r = sin(head_angle[2] * pi / 180)
        sin_y = sin(head_angle[0] * pi / 180)
        cos_y = cos(head_angle[0] * pi / 180)
        sin_p = sin(head_angle[1] * pi / 180)
        cos_p = cos(head_angle[1] * pi / 180)
        
        x = int((face_coords[0] + face_coords[2]) / 2)
        y = int((face_coords[1] + face_coords[3]) / 2)
        
        cv2.line(image, (x,y), (x+int(70*(cos_r*cos_y+sin_y*sin_p*sin_r)), y+int(70*cos_p*sin_r)), (255, 0, 0), 2)
        cv2.line(image, (x, y), (x+int(70*(cos_r*sin_y*sin_p+cos_y*sin_r)), y-int(70*cos_p*cos_r)), (0, 0, 255), 2)
        cv2.line(image, (x, y), (x + int(70*sin_y*cos_p), y + int(70*sin_p)), (0, 255, 0), 2)
       
        return image

    def preprocess_output(self, image, outputs, face_coords, display):
        '''
        Preprocess the output before feeding it to the next model.
        '''
        
        head_angles  =  [outputs['angle_y_fc'][0][0], outputs['angle_p_fc'][0][0], outputs['angle_r_fc'][0][0]]
        
        if (display):
            out_image = self.draw_outputs(image,  head_angles, face_coords)
        
        return out_image,  head_angles