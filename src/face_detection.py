
import os
import cv2
from openvino.inference_engine import IECore
from util_function import  preprocess_input

class Model_face_detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU'):
        '''
        Set instance variables.
        '''
        self.device = device
        self.model = model_name
        self.model_structure = model_name
        self.model_weights = os.path.splitext(self.model_structure)[0] + ".bin"
        
        try:
            self.model = IECore().read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        
    def load_model(self):
        '''
        Load the model to the specified device.
        '''
        core = IECore()
        self.Network = core.load_network(network=self.model, device_name=self.device)
        return self.Network

    def predict(self, image, threshold, display):
        '''
        Run predictions on the input image.
        '''        
        p_frame = preprocess_input(image, self.input_shape)
   
        self.Network.start_async(request_id=0, inputs={self.input_name: p_frame})
        
        if self.Network.requests[0].wait(-1) == 0:
            
            outputs = self.Network.requests[0].outputs[self.output_name]
            out_image, face, coords = self.preprocess_output(image, outputs, threshold, display)
             
        return out_image, face, coords

    def preprocess_output(self, image, outputs, threshold, display):
        '''
        Preprocess the output before feeding it to the next model.
        '''
           
        coords = []
        face = image
        for box in outputs[0][0]:
        
            if box[2] >= threshold:
                
                xmin = int(box[3] * image.shape[1])
                ymin = int(box[4] * image.shape[0])
                xmax = int(box[5] * image.shape[1])
                ymax = int(box[6] * image.shape[0])
                coords.append(xmin)
                coords.append(ymin)
                coords.append(xmax)
                coords.append(ymax)
                face = image[ymin:ymax, xmin:xmax]
                if(display):
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)   
        
        return image, face, coords