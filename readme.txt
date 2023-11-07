YOLO powered vechile detection implemented with PyTorch and utilizes Nvidia CUDA for GPU acceleration.
AMD users will default to CPU computation and experience worse performance.

### INSTALLATION ###

1. Install Nvidia CUDA if applicable
	https://developer.nvidia.com/cuda-downloads
	
2. Create and activate Pyrthon virtual environment
	python -m venv myenv
	myenv\Scripts\activate
	
3. Install requirements
	pip install -r requirements.txt
	
4. Launch Jupyter Notebook
	jupyter notebook
	
5. Run screencap.ipynb in Jupyter Notebook

## USAGE ## 

Drag window with title bar
Resize window by dragging bottom right corner
Position transparent window over video footage
Press Green "Start" button to start detection
Press Green "Draw Line" button and left-click and drag in capture window to draw counter line.
Counted vechiles displayed on the right.
Close window with top-right 'X'

Second window displaying detections provided, reduce capture window region to increase model performance (FPS)
