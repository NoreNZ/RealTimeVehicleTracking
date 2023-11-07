# RealTimeVehicleTracking
Real-time vehicle tracking in Python, powered by PyTorch and Yolov3n

YOLOv3n is implemented with PyTorch and utilizes Nvidia CUDA for GPU acceleration.
AMD users/users without CUDA installed will default to CPU computation and experience worse performance.

![ezgif com-resize](https://github.com/NoreNZ/RealTimeVehicleTracking/assets/50392938/a9a9abae-885b-4e16-93e9-b6caec9319ad)


### INSTALLATION ###

1. Install Nvidia CUDA if applicable
   
	https://developer.nvidia.com/cuda-downloads
	
2. Create and activate Python virtual environment
   
	python -m venv myenv

	myenv\Scripts\activate
	
3. Install requirements
   
	pip install -r requirements.txt

5.1 Run via Jupyter notebook

	jupyter notebook
	
5.2 or run via python script

  python run.py

## USAGE ## 

Drag window with title bar
Resize window by dragging bottom right corner
Position transparent window over video footage
Press Green "Start" button to start detection
Press Green "Draw Line" button and left-click and drag in capture window to draw counter line.
Counted vechiles displayed on the right.
Close window with top-right 'X'

Second window displaying detections provided, reduce capture window region to increase model performance (FPS)

![ezgif com-resize (1)](https://github.com/NoreNZ/RealTimeVehicleTracking/assets/50392938/759a9bac-656a-4a73-94dc-ca0267d0632c)
