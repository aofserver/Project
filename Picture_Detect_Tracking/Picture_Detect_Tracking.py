# python -m pip install https://files.pythonhosted.org/packages/0e/ce/f8a3cff33ac03a8219768f0694c5d703c8e037e6aba2e865f9bae22ed63c/dlib-19.8.1-cp36-cp36m-win_amd64.whl#sha256=794994fa2c54e7776659fddb148363a5556468a6d5d46be8dad311722d54bfcf
# pip install dlib

import cv2
import numpy as np
import dlib
import statistics

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

ID_list = {}
# mask_intersection
mask_intersection1 = 0
mask_intersection2 = 0

subtractor = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=100, detectShadows=True)
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
img_add = []
imgsub = 0
cnt = 0
Class = ""
listClass = {}


while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if cnt == 0:
        sh = frame.shape
        mask_intersection1 = np.ones((sh[0], sh[1])) * 0
        mask_intersection2 = np.ones((sh[0], sh[1])) * 0
        cnt = cnt + 1

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    mask_img = subtractor.apply(frame)
    if len(img_add) < 5:
        img_add.append(mask_img)
    if len(img_add) == 5:
        imgsub = img_add[0] + img_add[1] + img_add[2] + img_add[3] + img_add[4]
        img_add.pop(0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        ID_list = {}
        mask_intersection1 = mask_intersection1 * 0
        mask_intersection2 = mask_intersection2 * 0
        mask_result = mask_intersection1 * mask_intersection2

    for face in range(len(faces)):
        x1 = faces[face].left()
        y1 = faces[face].top()
        x2 = faces[face].right()
        y2 = faces[face].bottom()


        ###########  classification  ############
        cntsubimg = cv2.countNonZero(fgmask[y1:y2, x1:x2])
        if ((cntsubimg) / (abs(y1 - y2) * abs(x1 - x2))) > 0.7:
            Class = "Picture"
        else:
            Class = "Person"



        ###########  tracking  ############
        ID = face
        list_key = ID_list.keys()
        list_key = list(list_key)
        if ID in list_key:
            if len(faces) > 0:
                max_intersection = 0
                for i in range(len(ID_list)):
                    mask_intersection1 = mask_intersection1 * 0
                    mask_intersection2 = mask_intersection2 * 0
                    mask_intersection1[ID_list[i][1]:ID_list[i][3], ID_list[i][0]:ID_list[i][2]] = 1
                    cv2.imshow("mask_intersetion1", mask_intersection1)
                    mask_intersection2[y1:y2, x1:x2] = 1

                    mask_result = mask_intersection1 * mask_intersection2
                    if cv2.countNonZero(mask_result) > max_intersection:
                        max_intersection = cv2.countNonZero(mask_result)
                        ID_list[i] = [x1, y1, x2, y2,Class]

                        if len(listClass[i]) >= 9:
                            listClass[i].pop(0)
                            listClass[i].append(Class)
                            Class = statistics.mode(listClass[i])
                        else:
                            listClass[i].append(Class)

                        if Class == "Picture":
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, "ID "+str(i)+" "+Class, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(frame, "ID " + str(i) + " " + Class, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        else:
           ID_list[ID] = [x1,y1,x2,y2,Class]
           listClass[ID] = [Class]
           if Class == "Picture":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "ID "+str(ID)+" "+Class, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
           else:
               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
               cv2.putText(frame, "ID " + str(ID) + " " + Class, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



    cv2.imshow("Frame", frame)
    cv2.imshow("mask_intersetion2", mask_intersection2)


    key = cv2.waitKey(1)
    if key == 27:
        break