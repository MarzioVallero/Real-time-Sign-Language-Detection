import cv2
import numpy as np
import dlib
from math import hypot
from numpy import ndarray

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(".\shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)

        # gaze detection
        # seleziono l'area che comprende l'occhio sx
        # di base se un occhio guarda da un lato, anche l'altro guarda nella stessa direzione
        # quindi lavoriamo su un solo occhio
        left_eye_region: ndarray = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                             (landmarks.part(37).x, landmarks.part(37).y),
                                             (landmarks.part(38).x, landmarks.part(38).y),
                                             (landmarks.part(39).x, landmarks.part(39).y),
                                             (landmarks.part(40).x, landmarks.part(40).y),
                                             (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
        # questa riga disegna un poligono usando come punti quelli identificati prima
        # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

        # creo una maschera che comprenda tutta l'immagine tranne l'occhio
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8) # a questo punto è un riquadro che comprende tutta l'immagine

        # rimozione della zona dell'occhio dalla maschera appena creata
        cv2.polylines(mask, [left_eye_region], True, 255, 2) # bordo occhio
        cv2.fillPoly(mask, [left_eye_region], 255) # riempimento occhio

        # vogliamo isolare l'occhio

        # usando le maschere, isoliamo solo la parte dell'occhio
        left_eye = cv2.bitwise_and(gray, gray, mask=mask)

        # identifichiamo quindi un riquadro che lo contenga scegliendo i valori max e min sugli assi x e y
        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])

        # creazione del riquadro per l'occhio a partire dai punti max e min
        eye = frame[min_y: max_y, min_x: max_x]
        # rendo l'occhio in bianco e nero per poi fare il thresholding
        gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY) # thresholding tra 50 e 255

        # divido il threshold in parte sinistra e destra
        height, width = threshold_eye.shape
        # lato sx: da 0 a max altezza, da 0 a metà larghezza
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        # lato dx: da 0 a max altezza, da metà larghezza a 0
        right_side_threshold = threshold_eye[0: height, int(width / 2): width]

        # avendo isolato le due parti, possiamo vedere se c'è più bianco nella parte sx o dx
        # salviamo la quantità di pixel bianchi (il nero è visto come zero, quindi contiamo i non zero) nei due threshold
        # ho aggiunto 1 perché altrimenti all'apertura del programma le due variabili hanno valore 0
        # facendo invalidare gaze_ratio
        left_side_white = cv2.countNonZero(left_side_threshold) + 1
        right_side_white = cv2.countNonZero(right_side_threshold) + 1

        # per capire la direzione dello sguardo, confronto i due valori appena trovati
        # se sono simili sto guardando al centro, altrimenti a sx o dx
        # per confrontarli scelgo di calcolarne il rapporto
        gaze_ratio = left_side_white/right_side_white

        # stampe usate per verificare sperimentalmente i valori per left, center e rigth
        cv2.putText(frame, str(left_side_white), (50, 100), font, 2, (0, 0, 255), 3)
        cv2.putText(frame, str(right_side_white), (50, 150), font, 2, (0, 0, 255), 3)
        cv2.putText(frame, str(gaze_ratio), (50, 200), font, 2, (0, 0, 255), 3)

        if gaze_ratio <= 0.91:
            cv2.putText(frame, "RIGHT", (400, 100), font, 2, (0, 0, 255), 3)
        # new_frame[:] = (0, 0, 255)
        elif 0.91 < gaze_ratio < 1.1:
            cv2.putText(frame, "CENTER", (400, 100), font, 2, (0, 0, 255), 3)
        else:
            # new_frame[:] = (255, 0, 0)
            cv2.putText(frame, "LEFT", (400, 100), font, 2, (0, 0, 255), 3)

        cv2.imshow("EYE", eye)
        threshold_eye=cv2.resize(threshold_eye, None, fx=10, fy=10)
        cv2.imshow("EYE_threshold", threshold_eye)
        cv2.imshow("mask", mask)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()