import socket
import cv2
import numpy
import struct  # to send `int` as  `4 bytes`
import time  # for test
import threading
import queue

frames = queue.Queue(2)

TCP_IP = '10.0.0.13'
TCP_PORT = 5001
CAMERA_FPS = 120
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 272

class ImageGrabber(threading.Thread):
    def __init__(self, ID):
        threading.Thread.__init__(self)
        self.ID = ID
        self.cam = cv2.VideoCapture(ID)
        fps = self.cam.get(cv2.CAP_PROP_FPS)
        print("default fps is", fps)
        self.cam.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        fps = self.cam.get(cv2.CAP_PROP_FPS)
        print("Set fps to", fps)

    def run(self):
        global frames
        print("Thread running")
        while True:
            ret, frame = self.cam.read()

            frames.put(cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT)))
            time.sleep(.2)


################################################################################
#       Program main
################################################################################
print("RPI Send Camera image script Start")
sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
grabber = ImageGrabber(0)
grabber.start()
while True:

    if not frames.empty():
        Currframe = frames.get()
        result, imgencode = cv2.imencode('.jpg', Currframe, encode_param)
        data = numpy.array(imgencode)
        stringData = data.tostring()
       #print("image size is", len(stringData))
        print("queue size is ", frames.qsize(), "image size is", len(stringData))
        len_str = struct.pack('!i', len(stringData))

        sock.send(len_str)
        sock.send(stringData)
        #time.sleep(1)

sock.close()
