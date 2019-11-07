"""
Module for generating visual feature extraction.
"""
import cv2
from PIL import Image
from torch_mtcnn import detect_faces

"""
Currently, two methods for extracting bounding-boxes on faces have been tested,
a prebuilt mtcnn library and an OpenCV method based on Haar Cascades. So far,
mtcnn seems significally slower than the haar cascades method, but may be due
to the fact that it was run on cpu. Haar cascades is fast, but fails at detecting
side profiles and also produces false positives

"""

def detect_faces_mtcnn(video_tensor, display_images=False):
    """
    Method for facial extraction using a prebuilt mtcnn model
    """

    total_predictions = []

    for image in video_tensor:

        image_np = image.numpy()

        image_pil = Image.fromarray(image_np)
        predictions, landmarks = detect_faces(image_pil)

        print("Detected {} faces!".format(len(predictions)))
        total_predictions.append(predictions)

        if display_images:
            for (x, y, x2, y2, s) in predictions:
                #print(x, y, w, h)
                cv2.rectangle(image_np, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)

            cv2.imshow("Faces Found", image_np)
            cv2.waitKey(0)

    return total_predictions


def detect_faces_cascade(video_tensor, cascade_path, display_images=False):
    """
    Method for extracting facial extraction using an OpenCV implementation of
    the haar cascade algorithm

    Must provide a path to the cascade xml file.
    """
    faceCascade = cv2.CascadeClassifier(path)

    total_predictions = []

    for image in video_tensor:

        image_np = image.numpy()
        # Convert to grayscale
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        predictions = faceCascade.detectMultiScale(
            image_np,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        print("Detected {} faces!".format(len(predictions)))
        total_predictions.append(predictions)

        # For debugging purposes:
        if display_images:
            for (x, y, w, h) in predictions:
                cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Faces Found", image_np)
            cv2.waitKey(0)

    return total_predictions
