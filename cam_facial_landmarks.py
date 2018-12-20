# import the necessary packages
from imutils import face_utils
import dlib
import face_recognition
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt

def face_encoding_wrapper(image, upsample=1):
    try:
        return face_recognition.face_encodings(this_image)[0]

    except IndexError:
        if upsample != 4:
            return face_encoding_wrapper(upsample=4)
        raise
        
def train_from_examples():
    names = []
    faces = []
    img_files = glob("./examples/*.png")
    names = [f.split('/')[-1].split('.')[0] for f in img_files]
    for img_file in img_files:
        im = face_recognition.load_image_file(img_file)
        face_locations = face_recognition.face_locations(im)
        print("Found {} faces in image.".format(len(face_locations)))
        face_encodings = face_recognition.face_encodings(im, face_locations)
        faces += face_encodings

    return names, faces

def test_bp_image_encoding():
    p = "install-dlib-example/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    img_files = glob("./examples/*.png")
    names = [f.split('/')[-1].split('.')[0] for f in img_files]
    for img_file in img_files:
        im = face_recognition.load_image_file(img_file)
        marked_im = add_landmarks_to_image(im, detector, predictor)
        plt.imshow(im)
        plt.show()
        

def add_landmarks_to_image(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    return image
    
def main():
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    p = "data/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    cap = cv2.VideoCapture(0)

    print(\
    """
    Press '<Space>' to take a photo.
    Press 'c' to close the annotated face window.
    Press 'q' when in video camera mode to exit.
    """)

    while True:
        ret, frame = cap.read()
        cv2.imshow("Video Feed", frame)
        press = cv2.waitKey(1)

        if press & 0xFF == ord('c'):
            cv2.destroyWindow("Output")
            continue

        if press & 0xFF == ord('q'):
            break

        if press & 0xFF == ord(' '):
            marked_frame = add_landmarks_to_image(frame, detector, predictor)
            

            cv2.imshow("Output", marked_frame)
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
