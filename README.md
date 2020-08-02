# Computer Pointer Controller

In this project, we will be controlling computer pointer with head pose angles and eye gaze. This is done by using the pretrained models available at Intel Open Model Zoo. The diagram below shows the flow of data

![workflow-diagram](images/pipeline.png)

## Project Set Up and Installation

- Setup Intel OpenVINO™ Toolkit on your PC

- Install the `virtualenv` package. Then create the Virtual Environment and Activate it.
  ```
  pip install virtualenv
  virtualenv venv
  source venv/bin/activate
  ```
- Set the Environment Variables by running the setupvars script
  ```
  source /opt/intel/openvino/bin/setupvars.sh
  ```
- Install required dependencies using the requirements.txt file
  ```
  pip install -r requirements.txt 
  ```
- Create a folder models and download the required models into that folder using the downloader.py that comes with OpenVINO™ toolkit
  ```
  mkdir models
  cd models
  sudo /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001
  sudo /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001
  sudo /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009
  sudo /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002
  
  ```

## Project directory structure

```bash
.
├── bin
│   └── demo.mp4
├── images
│   └── pipeline.png
├── models
│   └── intel
│       ├── face-detection-adas-binary-0001
│       ├── gaze-estimation-adas-0002
│       ├── head-pose-estimation-adas-0001
│       └── landmarks-regression-retail-0009
├── src
│   ├── app.py
│   ├── face_detection.py
│   ├── facial_landmarks_detection.py
│   ├── gaze_estimation.py
│   ├── head_pose_estimation.py
│   ├── input_feeder.py
│   └── mouse_controller.py
├── README.md
└── requirements.txt
```

## Demo

To run the application in the webcam mode no arguments are needed just run main file without any parmeters: python main.py
To run using the demo video follow the Documentation section. we are going to use [-fd][-hp][-fl][-ge][-i] arguments and the command as follows

  ```
  python src/main.py -fd models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -hp models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -fl models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -ge models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i bin/demo.mp4
  ```

## Documentation

Supported Command line arguments as follows

```
usage: app.py [-h] [-m_FD FACE_DETECTION] [-m_HP HEAD_POSE_ESTIMATION]
              [-m_LM FACIAL_LANDMARKS_DETECTION] [-m_GE GAZE_ESTIMATION]
              [-i INPUT] [-it INPUT_TYPE] [-d DEVICE]
              [--extensions EXTENSIONS] [-pt PROB_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -m_FD FACE_DETECTION, --face_detection FACE_DETECTION
                        Path to pre-trained Face Detection model
  -m_HP HEAD_POSE_ESTIMATION, --head_pose_estimation HEAD_POSE_ESTIMATION
                        Path to pre-trained Head Pose Estimation Model
  -m_LM FACIAL_LANDMARKS_DETECTION, --facial_landmarks_detection FACIAL_LANDMARKS_DETECTION
                        Path to pre-trained Facial Landmarks Detection Model
  -m_GE GAZE_ESTIMATION, --gaze_estimation GAZE_ESTIMATION
                        Path to pre-trained Gaze Estimation Model
  -i INPUT, --input INPUT
                        Input File Path
  -it INPUT_TYPE, --input_type INPUT_TYPE
                        Input type: video or cam
  -d DEVICE, --device DEVICE
                        Device to run iinference on: Default CPU.
  --extensions EXTENSIONS
                        Any extensions for the selected device
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Thershold value for filtering: Default:0.5
```

## Benchmarks

Benchmarks on CPU Device for different precision levels.

FP32:
	model load time = 0.58
	inference time = 12.00
FP16:
	model load time = 0.62
	inference time = 11.82
INT8:
	model load time = 0.87
	inference time = 11.68

## Results

Based on the benchmark results we came to conclusion that lowering the precision ends up in lower accuracy. However it increases the inference rate.
