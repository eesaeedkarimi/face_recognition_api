Overview
--------
This is an api for Face Recognition. The module uses [InsightFace](https://github.com/deepinsight/insightface) for
detection and recognition of faces. Each endpoint of this api is responsible for a task, including Identification of an 
unknown face, Enrollment of a face in database, Healthcheck, etc.

User Guide
----------

**Running API**

Run the following commands to start the api. In this example the api will listen to port 6000 of local system. You
can send your requests to this port to get a response as below:

    $ pip install requirements.txt
    $ python api.py --host 127.0.0.1 --port 6000

If you want to use this api in a production service run the following commands to build a docker-compose:

    $ sudo docker-compose build
    $ sudo docker-compose up -d


**Using Face Identification endpoint of API**

In order to identify a person by their face you can access to this api and send your requests to the following endpoint:

    http://127.0.0.1:6000/face_recognition/face_identification

The request should include A base64 encoded of the frame, an id for the request, and an integer of 0, 1, or 2
that defines 90, 180, 270 degrees of the rotation of the frames. The response would be the possible "error_message",
a boolean showing if the "face_is_detected", a boolean showing if the "face_is_identified", an integer between 0 and 100
showing the "face_similarity" between detected face and the enrolled face of the identified person, "identified_id",
and "identified_name".

**Input JSON Structure of Face Identification**

    {
        "frame": "A base64 encoded of the frame",
        "request_id": "request id",
        "rotation": "An integer of 0, 1, or 2 that defines 90, 180, 270 degrees of rotation",
    }

**Example Request for Face Identification**

To generate the **json_frames**, please follow the codes:

> cv_frame = cv2.imread(os.path.join(BASE_ADDRESS, "tests/fixtures/hodor_2.jpg"))  
> frame_str_identification = base64.b64encode(np.array(cv2.imencode(".jpg", cv_frame)[1]).tobytes()).decode("utf8")

    POST 127.0.0.1:6000/face_recognition/face_identification HTTP/1.1

    {
        "frame": frame_str_identification,
        "request_id": "test_face_identification",
        "rotation": "0",
    }


**Using Face Enrollment endpoint of API**

In order to enroll a person in the database you can access to this api and send your requests to the following endpoint:

    http://127.0.0.1:6000/face_recognition/face_enrollment

The request should include a base64 encoded of the frame, a positive integer that defines the id of user to be enrolled,
a string that defines the name of user to be enrolled, an id for the request, and an integer of 0, 1, or 2 that defines
90, 180, 270 degrees of the rotation of the frames. The response would be the possible "error_message", the
"enrolled_id", and the "enrolled_name".

**Input JSON Structure of Face Enrollment**

    {
        "frame": "A base64 encoded of the frame",
        "user_id": "A positive integer that defines the id of user to be enrolled",
        "user_name": "An string that defines the name of user to be enrolled",
        "request_id": "request id",
        "rotation": "An integer of 0, 1, or 2 that defines 90, 180, 270 degrees of rotation",
    }

**Example Request for Face Enrollment**

To generate the **json_frames**, please follow the codes:

> cv_frame = cv2.imread(os.path.join(BASE_ADDRESS, "tests/fixtures/hodor_1.jpg"))  
> frame_str_enrollment = base64.b64encode(np.array(cv2.imencode(".jpg", cv_frame)[1]).tobytes()).decode("utf8")

    POST 127.0.0.1:6000/face_recognition/face_enrollment HTTP/1.1

    {
        "frame": frame_str_enrollment,
        "user_id": 43,
        "user_name": "hodor",
        "request_id": "test_face_enrollment",
        "rotation": "0",
    }


**Using Health Check endpoint of API**

In order to check if the service is available you can access to this api and send your requests to the following endpoint:

    http://127.0.0.1:6000/face_recognition/health_check

The request should be a get request without any parameters. The response would be a dictionary with a "status" key that 
its value is "Green".


**Example Request for Health Check**

    GET 127.0.0.1:6000/face_recognition/face_enrollment HTTP/1.1
