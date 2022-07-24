import base64

import cv2
import numpy as np
import requests

url = "http://192.168.1.107:6000/face_recognition/face_identification"

################################################################
# Show settings
font = cv2.FONT_HERSHEY_SIMPLEX
org = (1, 20)
fontScale = 0.75
identified_color = (0, 255, 0)
unidentified_color = (0, 0, 255)
text_thickness = 2
face_rectangle_thickness = 2

# define video capture object
cap = cv2.VideoCapture(0)
frame_number = 0

while True:

    ret, cv_frame = cap.read()
    frame_str_identification = base64.b64encode(np.array(cv2.imencode(".jpg", cv_frame)[1]).tobytes()).decode("utf8")
    payload = {
        'frame': frame_str_identification,
        'request_id': f'live_test_frame_{frame_number}',
        }
    response = requests.request("POST", url, headers={}, data=payload, files=[])
    data = response.json()
    if ("face_is_detected" in data) & (data["face_is_detected"]):
        if data["face_is_identified"]:
            color = identified_color
            # text = f'Identified ID: {data["identified_id"]}, Identified Name: {data["identified_name"]}'
            text = f'Welcome {data["identified_name"]}, ID: {data["identified_id"]}'
            cv_frame = cv2.putText(cv_frame, text, org, font, fontScale, color, text_thickness, cv2.LINE_AA)
        else:
            color = unidentified_color
        # top left
        start_point = (int(data["bbox"]["left"]), int(data["bbox"]["top"]))
        # bottom right
        end_point = (int(data["bbox"]["right"]), int(data["bbox"]["bottom"]))
        # Draw a rectangle with color line borders of face_rectangle_thickness
        cv_frame = cv2.rectangle(cv_frame, start_point, end_point, color, face_rectangle_thickness)

    print(response.text)

    frame_number += 1
    # Display the resulting frame
    cv2.imshow('frame', cv_frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
