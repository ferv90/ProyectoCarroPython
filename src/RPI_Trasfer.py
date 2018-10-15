import socket
import cv2
import numpy
import struct # to send `int` as  `4 bytes`
import time   # for test


TCP_IP = '10.0.0.13'
TCP_PORT = 5001

print("inicio del programa")
sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = data.tostring()
    print("image size is", len(stringData))

    len_str = struct.pack('!i', len(stringData))

    sock.send(len_str)
    sock.send(stringData)
    time.sleep(10)

sock.close()

