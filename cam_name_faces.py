from imutils import face_utils
import subprocess
import threading
import time
import dlib
import json
import face_recognition
import datetime
import numpy as np
import cv2
from glob import glob
import os
import matplotlib.pyplot as plt
import random
import argparse
import pickle

TRAINED_FACES_FILE = 'data/face_model.pckl'
TOLERANCE = 0.6
NAMES_TO_MUSIC = 'data/names2music.json'
MAX_SONG_DUR = 30 # seconds
NEW_PERSON_GREETING = 'music/GSFRIEND.mp3'

global key_press

class MusicPlayer(threading.Thread):
    def __init__(self):
        pass

    def run(self):
        pass

class KnownFaces:
    def __init__(self, names, encodings):
        assert len(names) == len(encodings)
        self.names = names
        self.encodings = encodings

    def match2name(self, match_list):
        assert len(self.names) == len(match_list)

        # Could return empty list, list of length 1, or greater
        return [name for (name, match) in zip(self.names, match_list) if match]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", help="Retrain known faces from examples folder", action="store_true")

    return parser.parse_args()

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

    model = KnownFaces(names, faces)
    model.save(TRAINED_FACES_FILE)
            
    return model
        
def add_landmarks_to_image(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    return image


def call_vlc(song_file):
    return subprocess.Popen(['vlc', song_file, '--intf', 'dummy'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def stop_songs(players):
    [p.kill() for p in players]

def play_songs(names):
    themesong = load_json(NAMES_TO_MUSIC)
    songs2play = [themesong.get(n, NEW_PERSON_GREETING) for n in names]
    start_time = datetime.datetime.now()
    players = [call_vlc(s) for s in songs2play]

    now = datetime.datetime.now()
    while (now - start_time) < datetime.timedelta(seconds=MAX_SONG_DUR):
        if key_press & 0xFF == ord('c'):
            break;

    stop_songs(players)

def detect_faces(image, known):
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    face_names = []

    for fe in face_encodings:
        matches = face_recognition.compare_faces(known.encodings, fe, tolerance=TOLERANCE)
        if sum(matches) > 0:
            name = random.choice(known.match2name(matches))
        else:
            name = "<Unknown Person>"

        face_names.append(name)

    return face_names, face_locations

def display_faces(image, face_names, face_locations):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue
        
        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Image has now been modified
    
    return image

        
def annotate_image(image, known):
    face_names, face_locations = detect_faces(image, known)
    return display_faces(image, face_names, face_locations)
    
def main():
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    global key_press
    args = parse_args()

    if args.train:
        known = train_from_examples()
    else:
        known = KnownFaces.load(TRAINED_FACES_FILE)
        
    # p = "data/shape_predictor_68_face_landmarks.dat"
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(p)
    # known = train_from_examples()

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
        key_press = cv2.waitKey(1)

        # Limit effects of "key bouncing"
        time.sleep(0.05)
        

        if key_press & 0xFF == ord('c'):
            cv2.destroyWindow("Output")
            continue

        if key_press & 0xFF == ord('q'):
            break

        if key_press & 0xFF == ord(' '):
            # marked_frame = add_landmarks_to_image(frame, detector, predictor)
            # marked_frame = annotate_image(frame, known)
            face_names, face_locations = detect_faces(frame, known)
            marked_frame = display_faces(frame, face_names, face_locations)
            cv2.imshow("Output", marked_frame)
            # This will cause some errors/confusion
            t = threading.Thread(target=play_songs, args=(face_names,))
            t.start()

            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
