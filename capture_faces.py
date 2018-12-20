import cv2
import os

def store_name(out_dir, img):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    name = input("Who is this? [enter 'retake' to retake]\n")
    if name.lower() == 'retake':
        pass
    else:
        filename = os.path.join(out_dir, name + '.png')
        cv2.imwrite(filename, img)
        print("You have been saved!")

def main():
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
            store_name('examples', frame)
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


