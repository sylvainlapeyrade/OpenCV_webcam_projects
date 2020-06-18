import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar

video_capture = cv2.VideoCapture(0)
hasFrame, frame = video_capture.read()

# Record the test
# vid_writer = cv2.VideoWriter('./demo/demo.avi', cv2.VideoWriter_fourcc(
#     'M', 'J', 'P', 'G'), 10, (frame.shape[1], frame.shape[0]))


# Display barcode and QR code location
def display(im, decodedObjects):

    # Loop over all decoded objects
    for decodedObject in decodedObjects:
        points = decodedObject.polygon

        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(
                np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points

        # Number of points in the convex hull
        n = len(hull)

        # Draw the convext hull
        for j in range(0, n):
            cv2.line(im, hull[j], hull[(j+1) % n], (255, 0, 0), 3)

    # Display results
    # cv2.imshow("Results", im);


# Create a qrCodeDetector Object
qrDecoder = cv2.QRCodeDetector()

# Detect and decode the qrcode
while(1):
    hasFrame, inputImage = video_capture.read()

    if not hasFrame:
        break

    decodedObjects = pyzbar.decode(inputImage)
    if len(decodedObjects):
        zbarData = decodedObjects[0].data
    else:
        zbarData = ''

    opencvData, bbox, rectifiedImage = qrDecoder.detectAndDecode(inputImage)

    if zbarData:
        cv2.putText(img=inputImage, text="ZBAR : {}".format(zbarData),
                    org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75, color=(0, 255, 0), thickness=2,
                    lineType=cv2.LINE_AA)
    else:
        cv2.putText(img=inputImage, text="ZBAR : QR Code NOT Detected",
                    org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75, color=(0, 0, 255), thickness=2,
                    lineType=cv2.LINE_AA)
    if opencvData:
        cv2.putText(img=inputImage, text="OpenCV: {}".format(opencvData),
                    org=(10, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75, color=(0, 255, 0), thickness=2,
                    lineType=cv2.LINE_AA)
    else:
        cv2.putText(img=inputImage, text="OpenCV : QR Code NOT Detected",
                    org=(10, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75, color=(0, 0, 255), thickness=2,
                    lineType=cv2.LINE_AA)

    display(inputImage, decodedObjects)
    cv2.imshow("Result", inputImage)
    # vid_writer.write(inputImage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release all ressources
cv2.waitKey(0)
cv2.destroyAllWindows()
video_capture.release()
# vid_writer.release()
