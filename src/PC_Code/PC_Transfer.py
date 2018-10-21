import socket
import cv2
import numpy
import struct
import time  # for test
import os
import sys
sys.path.append(os.path.abspath("/Users/fer/Documents/ProyectoCarroPython/ProyectoCarroPython/src/PC_Code"))
from ObjectDetection import *


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


TCP_IP = '169.254.7.200'
TCP_PORT = 5001

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", TCP_PORT))
s.listen(True)
conn, addr = s.accept()
Object = ObjectDetect()
try:
    while True:
        len_str = recvall(conn, 4)
        size = struct.unpack('!i', len_str)[0]
        # print("the length image is ", size)

        stringData = recvall(conn, int(size))
        data = numpy.frombuffer(stringData, dtype='uint8')
        decimg = cv2.imdecode(data, 1)
        start = time.time()
        decimg = Object.ProcessImage(decimg)
        print("Time execution is ",time.time() - start)
        cv2.imshow('Captura RPI', decimg)
        cv2.waitKey(1)

except KeyboardInterrupt:
    cv2.destroyAllWindows()
    s.close()
    pass
