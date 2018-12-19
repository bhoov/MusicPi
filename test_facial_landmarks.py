# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import matplotlib.pyplot as plt

def add_landmarks_to_image(image):
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
    p = "install-dlib-example/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow("Video Feed", frame)
        main_press = cv2.waitKey(1)

        if main_press & 0xFF == ord('c'):
            cv2.destroyWindow("Output")
            continue

        if main_press & 0xFF == ord(' '):
            marked_frame = add_landmarks_to_image(frame)

            cv2.imshow("Output", marked_frame)
            press = cv2.waitKey(1)
            if press & 0xFF == ord('q'):
                break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    

