import socket
import cv2
import numpy
import struct

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


TCP_IP = '169.254.7.200'
TCP_PORT = 5001

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", TCP_PORT))
s.listen(True)
conn, addr = s.accept()
try:
    while True:
        len_str = recvall(conn, 4)
        size = struct.unpack('!i', len_str)[0]
        #print("the length image is ", size)

        stringData = recvall(conn, int(size))
        data = numpy.frombuffer(stringData, dtype='uint8')

        decimg = cv2.imdecode(data, 1)
        cv2.imshow('Captura RPI', decimg)
        cv2.waitKey(1)

except KeyboardInterrupt:
    cv2.destroyAllWindows()
    s.close()
    pass
