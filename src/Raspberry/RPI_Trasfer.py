import socket
import cv2
import numpy
import struct  # to send `int` as  `4 bytes`
import time  # for test
import threading
import queue
from picamera.array import PiRGBArray
from picamera import PiCamera

frames = queue.Queue(2)

TCP_IP = '169.254.222.71'
TCP_PORT = 5001
CAMERA_FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

class ImageGrabber(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.camera = PiCamera()
        self.camera.resolution = (CAMERA_WIDTH, CAMERA_HEIGHT)
        self.camera.framerate = CAMERA_FPS
        self.camera.start_preview()
        time.sleep(2)
        self.rawCapture = PiRGBArray(self.camera, size=(CAMERA_WIDTH, CAMERA_HEIGHT))

    def run(self):
        global frames
        print("Thread running")
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr"):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            image = frame.array

            # clear the stream in preparation for the next frame
            self.rawCapture.truncate(0)

            frames.put(image)
            time.sleep(.1)


################################################################################
#       Program main
################################################################################
print("RPI Send Camera image script Start")
sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
grabber = ImageGrabber()
grabber.start()
while True:

    if not frames.empty():
        Currframe = frames.get()
        result, imgencode = cv2.imencode('.jpg', Currframe, encode_param)
        data = numpy.array(imgencode)
        stringData = data.tostring()
        #print("image size is", len(stringData))
        #print("queue size is ", frames.qsize(), "image size is", len(stringData))
        len_str = struct.pack('!i', len(stringData))
        start = time.time()
        sock.send(len_str)
        sock.send(stringData)
        print("Time execution is ",time.time() - start)

        #time.sleep(1)

sock.close()
