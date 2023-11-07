import sys
import threading
import os
import time

import cv2
import numpy as np
import torch
import mss
from collections import deque

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPainter, QFont, QColor, QImage, QPen, QPolygon
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QDesktopWidget

from deep_sort_realtime.deepsort_tracker import DeepSort




class MainWindow(QMainWindow):
    close_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.drag_start_position = None
        self.resize_start_position = None
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground) # Transparent window
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint # Always draw window on top of others
            | QtCore.Qt.X11BypassWindowManagerHint # Bypass windows manager
            | QtCore.Qt.FramelessWindowHint # Remove window frame
        )
        self.setGeometry(100, 100, 400, 200)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0);")

        self.beginbutton = False # Button bool state for starting object detection

        # Define rects for different UI components.
        self.capturerect = QtCore.QRect(0, 25, self.width()-200, self.height()-25)
        self.confirmrect = QtCore.QRect(round(int(self.capturerect.width()/2-50)), round(int(self.capturerect.height()/2-10)), 100,50)
        self.toprect = QtCore.QRect(
            0,
            0,
            self.width() - 1,
            25)
        self.quitrect = QtCore.QRect(
            self.width()-25,
            0,
            25,
            25)
        self.rightrect = QtCore.QRect(
            self.width()-150,
            25, 
            self.width(), 
            self.height())
        self.drawlinerect = QtCore.QRect(
            self.rightrect.x()+5,
            self.rightrect.y() + self.rightrect.height()-75,
            140,
            25
        )

        self.line_click_start = None
        self.line_click_end = None
        self.draw_line = False

        # Dictionary for storing class detections that have intersected counter line
        self.classcounter = {}
        # Counter for total detections that have intersected
        self.counter = 0

        self.model = torch.hub.load('ultralytics/yolov3', 'yolov5n') # Load yolov3n model
        if torch.cuda.is_available():
            device = torch.device('cuda')  # Use GPU acceleration
            print("Using CUDA-enabled GPU for acceleration.")
        else:
            device = torch.device('cpu')   # Use CPU
            print("No compatible CUDA-enabled GPU detected. Using CPU for computation.")


        # Use the device for the model
        self.model.to(device)
        self.model.classes = [1,2,3,5,7] # Restrict detection classes to vechiles
        self.model.conf = 0.1 # Reduced minimum confidence for detections, incorrect detections handled by other means.
        self.model.agnostic = True
        self.model.amp = False

    def resizeEvent(self, event):
        """
         Update the size of the capture region when the window is resized. This is a hack to avoid the issue where Qt doesn't know how to resize the window when the window resizes.
         
         @param event - The resize event that triggered this function call.
        """
        # Update the size of the capture region when the window is resized
        # Subtract 200 from with 
        self.capturerect = QtCore.QRect(0, 25, self.width() - 200, self.height() - 25)
        event.accept()

    def paintEvent(self, event):
        """
         PyQt Paint event. When called draws UI elements
         
         Args:
         	 event: Event that triggered the paint event.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Update UI element regions (rects)
        rect = QtCore.QRect(0, 0, self.width() - 1, self.height() - 1)
        self.rightrect = QtCore.QRect(self.width()-150, 25, self.width(), self.height())
        self.capturerect = QtCore.QRect(0, 25, self.width()-150, self.height()-25)
        self.confirmrect = QtCore.QRect(round(int(self.capturerect.width()/2-50)), round(int(self.capturerect.height()/2-10)), 100,50)
        self.toprect = QtCore.QRect(0, 0, self.width() - 1, 25)
        self.quitrect = QtCore.QRect(self.width()-25, 0, 25, 25)
        self.drawlinerect = QtCore.QRect(
            self.rightrect.x()+5,
            self.rightrect.y() + self.rightrect.height()-75,
            140,
            25
        )

        # Draw Whole window rect
        painter.fillRect(rect, QtGui.QColor(100, 100, 100, 50))
        # Draw right handside counter region
        painter.fillRect(self.rightrect, QtGui.QColor(255, 255, 255, 150))
        # Draw opaque infill for right handside counter region
        painter.fillRect(
            self.rightrect.x() + 10,
            self.rightrect.y() + 10,
            130,
            self.rightrect.height() - 80,
            QtGui.QColor(255, 255, 255, 200)
        )      
        font = QFont('Decorative', 15)
        painter.setFont(font)
        # Draw text for total detections counter
        painter.drawText(self.rightrect.x()+20, self.rightrect.y() + 40, f"Total: {self.counter}")
        # If classcounter detections, draw text for each on new line
        if self.classcounter:
            y = 40
            for value in self.classcounter:
                y += 30  # Move to the next line
                text = f"{value.capitalize()} : {self.classcounter[value]}"
                painter.drawText(self.rightrect.x()+20, self.rightrect.y() + y, text)
        font = QFont('Decorative', 10)
        painter.setFont(font)
        rect_pen = QtGui.QPen(QtGui.QColor(255, 0, 0))

        pen_thickness = 3
        rect_pen.setWidth(pen_thickness)
        painter.drawRect(rect) # Draw window border

        painter.fillRect(self.capturerect, QtGui.QColor(255, 0, 0, 1)) # Draw capture region rect, 1 opacity to still receive mouseclicks
        painter.fillRect(self.toprect, QtGui.QColor(255, 255, 255, 255)) # Draw top window handle rect, for dragging window
        painter.fillRect(self.quitrect, QtGui.QColor(255, 255, 255, 255)) # Draw quit button rect
        # If draw_line is False button colour green.
        if self.draw_line == False:
            painter.fillRect(self.drawlinerect, QtGui.QColor(50, 180, 40, 255))
        # Else draw_line is True and user is currently drawing line. Button colour gray.
        else: 
            painter.fillRect(self.drawlinerect, QtGui.QColor(180, 180, 180, 255))
        painter.drawText(self.drawlinerect, Qt.AlignCenter, "Draw Line") # Draw text for drawline button
        painter.drawText(self.drawlinerect.x()+30, self.drawlinerect.y()+25, self.drawlinerect.width(), self.drawlinerect.height(), Qt.AlignCenter, "Resize") # Draw text to label resize corner
        painter.drawText(10, 18, "Capture Window") # Draw window title

        font = QFont('Decorative', 15)
        painter.setFont(font)

        triangle_size = 20
        # Define corner resize triangle points
        triangle = QPolygon([
            self.rect().bottomRight() - QPoint(triangle_size, 0),
            self.rect().bottomRight(),
            self.rect().bottomRight() - QPoint(0, triangle_size)
        ])
        painter.setBrush(QColor(255, 255, 255))
        painter.drawPolygon(triangle) # Draw resize triangle
        quit_x = int(self.quitrect.width() / 2) + self.width() - 30
        quit_y = int(self.quitrect.height() / 2) + 8
        painter.drawText(quit_x, quit_y, "X") # Draw quit window 'X'

        line_pen = QtGui.QPen(QtGui.QColor(0, 255, 0, 50))
        line_pen.setWidth(3)
        painter.setPen(line_pen)
        # Draw the line click on the painter.
        if self.line_click_end and self.line_click_start:
            painter.drawLine(self.line_click_start[0], self.line_click_start[1], self.line_click_end[0], self.line_click_end[1])

        # Draw the confirm button if the button is the beginbutton is True.
        if self.beginbutton == False:
            painter.fillRect(self.confirmrect, QtGui.QColor(0, 255, 0, 255))
            painter.setPen(QtGui.QColor(0, 0, 0))
            font = QFont('Decorative', 12)
            painter.setFont(font)
            painter.drawText(self.confirmrect, Qt.AlignCenter, "Start") # Draw Start button text

    def intersect(self, p1, q1, p2, q2):
        """
         Checks if two lines intersect. This is used to determine if a line is intersecting by looking at orientations of the lines.
         
         Args:
         	 p1: The start point of the first line.
         	 q1: The start point of the first line.
         	 p2: The end point of the second line.
         	 q2: The end point of the second line.
         
         Returns: 
         	 True if the lines intersect False otherwise.
        """
        def orientation(p, q, r): 
            """
             Returns 1 if p is clockwise 2 if counter - clockwise and 0 otherwise.
             
             Args:
             	 p: Coordinates of point to check. Must be in [ x y ] order.
             	 q: Coordinates of point to check. Must be in [ x y ] order.
             	 r: Coordinates of point to check. Must be in [ x y ] order.
             
             Returns: 
             	 orientation of point to check.
            """
            
            val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1])) 
            # Return 1 if val 0 or 2 if val 0
            if (val > 0): 
                return 1
            elif (val < 0): 
                return 2
            else: 
                return 0
        
        o1 = orientation(p1, q1, p2) 
        o2 = orientation(p1, q1, q2) 
        o3 = orientation(p2, q2, p1) 
        o4 = orientation(p2, q2, q1) 
    
        # Return True if intersect
        if ((o1 != o2) and (o3 != o4)): 
            return True
    
        return False

    def mousePressEvent(self, event):
        """
         This method is called when the user presses the mouse. It checks to see if the mouse is over a window component.
         Used to handle button clicks, window dragging, window resizing

         Args:
         	 event: The QMouseEvent that triggered the call
        """
        # This method is called when the user clicks on the left button.
        if event.button() == Qt.LeftButton:
            # Quitbutton detection
            if self.quitrect.contains(event.pos()):
                os._exit(0)
            # Resize button detection
            elif self.width() - 25 <= event.pos().x() <= self.width() and self.height() - 25 <= event.pos().y() <= self.height():
                self.resize_start_position = event.globalPos()
                event.accept()
            # Drag window detection
            elif self.toprect.contains(event.pos()):
                self.drag_start_position = event.globalPos() - self.frameGeometry().topLeft()
                event.accept()
            # Confirm button detection
            elif self.confirmrect.contains(event.pos()):
                self.beginbutton = True
                self.main_thread = threading.Thread(target=self.main_loop)
                self.main_thread.start()
                event.accept()
            # Drawline button detection when button state False
            elif self.drawlinerect.contains(event.pos()) and self.draw_line == False:
                self.draw_line = True
                event.accept()
            # Record left click release location for line drawing
            elif self.draw_line == True:
                self.line_click_start = (event.pos().x(), event.pos().y())
                print(f'[Line] Left click down at ({event.pos().x()}, {event.pos().y()})')
                event.accept()

    def mouseMoveEvent(self, event):
        """
         Handle mouse move events. This is called when the user drags the mouse over the window.
         
         Args:
         	 event: The QMouseEvent that triggered this call. See Qt documentation
        """
        # Resize the window if the event is a left button.
        if event.buttons() & Qt.LeftButton:
            # Resize the window to the start of the window.
            if self.resize_start_position:
                delta = event.globalPos() - self.resize_start_position
                new_size = self.size() + QtCore.QSize(delta.x(), delta.y()) # Resize the window to the new size
                if new_size.width() > 400 and new_size.height() > 200:  # Check for minimum dimensions
                    self.resize(new_size)
                    self.resize_start_position = event.globalPos()
                event.accept()
            elif self.drag_start_position:
                self.move(event.globalPos() - self.drag_start_position)
                event.accept()

    def mouseReleaseEvent(self, event):
        """
         Mouse release event for the canvas. This is called when the mouse button is released
         
         Args:
         	 event: The QMouseEvent that triggered
        """
        self.drag_start_position = None
        self.resize_start_position = None  # Reset the resize_start_position
        # If the line is drawn on the line.
        if self.draw_line:
            # line click release on line click
            if not self.drawlinerect.contains(event.pos()):
                self.line_click_end = (event.pos().x(), event.pos().y()) # Record line end point
                print(f'[Line] Left click release at ({event.pos().x()}, {event.pos().y()})')
                self.draw_line = False
                event.accept()

    def main_loop(self):
        """
         Main loop. Runs Yolov3n powered object detection using window capture region as video input.
        """
        with mss.mss() as sct:
            pts = [deque(maxlen=10) for _ in range(1000)] # Deque used to store recent detection center points, used to draw trailing path lines
            tracker = DeepSort(max_age=10) # Initialize DeepSort tracker for detection persistency
            detections = [] # Array for storing Yolo detections every frame
            track_status = {} # Dictionary for storing detections
            intersect_status = {} # Dictionary for storing intersect state for detections
            line1 = None
            line2 = None
            colors_dict = {} # Dictionary for storing assigning stored detection colors

            # Main loop
            while True:
                window_position = self.mapToGlobal(QtCore.QPoint(0, 0)) # Record current window position
                # Initialize capture window region
                monitor = {
                    "left": window_position.x(),
                    "top": window_position.y()+self.toprect.height(),
                    "width": self.capturerect.width(),
                    "height": self.capturerect.height(),
                }

                grabbedframe = np.array(sct.grab(monitor)) # Screen capture monitor contents in capture region

                t1 = time.time() # Time start for FPS counter

                height, width, _ = grabbedframe.shape

                # Record detection results
                results = self.model(grabbedframe)
                detections = results.pandas().xyxy[0]
                frame = results.render()[0]

                bbs = [((row['xmin'], row['ymin'], row['xmax'] - row['xmin'], row['ymax'] - row['ymin']), row['confidence'], row['name']) for index, row in detections.iterrows()]


                # Pass detections to deep sort tracker
                tracks = tracker.update_tracks(bbs, frame=frame)

                # Iterate through each detected object
                for track in tracks:
                    if not track.is_confirmed():
                        # If the track is not confirmed, continue to the next track
                        continue

                    # Get the track ID and convert it to ltrb format
                    track_id = int(track.track_id)
                    ltrb = track.to_ltrb()

                    detclass = track.det_class
                    conf = track.det_conf
                
                    # Generate a unique color for each track_id or retrieve the color if already generated
                    if track_id not in colors_dict:
                        colors_dict[track_id] = tuple(np.random.randint(0, 256, 3).tolist() + [255])
                    assigned_color = colors_dict[track_id]

                    if track_id not in track_status:
                        track_status[track_id] = 1
                    else:
                        track_status[track_id] += 1

                    if conf != None:
                        conf = str(round(conf,2))
                    # Draw rectangle and text on the frame
                    bbox = ltrb           
                    
                    # Center of detection bounding box
                    center = (int((bbox[0]) + (bbox[2]))/2, int((bbox[1]) + (bbox[3]))/2)
                    pts[track_id].append(center)
                    point1, point2 = None, None

                    for j in range(1, len(pts[int(track.track_id)])):
                        if pts[int(track.track_id)][j-1] is None or pts[int(track.track_id)][j] is None:
                            continue
                        thickness = max(1, int(np.exp(0.2 * j) * 2))
                        cv2.line(grabbedframe, (int(pts[int(track.track_id)][j][0]), int(pts[int(track.track_id)][j][1])), (int(pts[int(track.track_id)][j-1][0]), int(pts[int(track.track_id)][j-1][1])), assigned_color, thickness)
                        point1 = (pts[int(track.track_id)][0])


                        if pts[int(track.track_id)][j] != None:
                            point2 = (pts[int(track.track_id)][j])
                            
                    if point1 and point2:
                        if self.line_click_start and self.line_click_end:
                            line1 = self.line_click_start
                            line2 = self.line_click_end

                            line1 = (line1[0], line1[1]-25)
                            line2 = (line2[0], line2[1]-25)

                            if track_id not in intersect_status:
                                if (self.intersect(point1, point2, line1, line2)):
                                    intersect_status[track_id] = True

                    # For each tracked item that is being counted
                    for track in track_status:
                        # If item has been counted more than 30 times
                        if track_id in track_status and track_id in intersect_status:
                            if (track_status[track_id] >= 20 and intersect_status[track_id] == True):
                                intersect_status[track_id] = False
                                self.counter += 1
                                if detclass not in self.classcounter:
                                    self.classcounter[detclass] = 1
                                else:
                                    self.classcounter[detclass] += 1
                                print(self.classcounter)

                    cv2.rectangle(grabbedframe, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), assigned_color, 2)
                    cv2.putText(grabbedframe, "ID: " + str(track_id) + str(track_status[track_id]), (int(bbox[0]), int(bbox[1] - 10)) , cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                assigned_color, 2)
                    cv2.putText(grabbedframe, str(conf), (int(bbox[0]), int(bbox[3]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                                assigned_color, 2)
                    
                fps = 1./(time.time()-t1)
                cv2.putText(grabbedframe, "FPS: {:.2f}".format(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if line1 and line2:
                    cv2.line(grabbedframe, line1, line2, (0,255,0),2)

                cv2.imshow('OpenCV Detection Window', grabbedframe)

                self.update()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break



if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywindow = MainWindow()
    mywindow.close_signal.connect(mywindow.close)  # Connect the close_signal to the close method
    mywindow.show()

    app.exec_()


