import os

# Importing current direction for future usage
BASE_ADDRESS = os.path.dirname(os.path.realpath(__file__))

ENROLLED_EMBEDDING_ADDRESS = os.path.join(BASE_ADDRESS, 'enrolled_embeddings')

URL = "http://127.0.0.1:6000/"
TEST_DELAY_TIME = 2000

# Defining max scale size
MAX_IMAGE_SCALE = 500.
# Blue color in BGR
FACE_RECTANGLE_COLOR = (0, 0, 255)
# Line thickness of 1 px
FACE_RECTANGLE_THICKNESS = 2
