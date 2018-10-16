import numpy as np
import cv2

cap = cv2.VideoCapture(0)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    decimg = cv2.imdecode(imgencode, 1)

    # Display the resulting frame
    cv2.imshow('Captura RPI', decimg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()