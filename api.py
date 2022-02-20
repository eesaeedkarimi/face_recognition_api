import argparse
import base64
import sys
import time

from flask import Flask, jsonify, request
from flask_restful import Resource, Api

import utils
# Packages
from config import BASE_ADDRESS
from face_recognition import FaceRecognition
from log import logger

# Initializing the flask API
app = Flask(__name__, instance_path=BASE_ADDRESS)
api = Api(app)

# Initializing the logger object
logger = logger()

# Initializing Face Identification Object
face_recognition = FaceRecognition()


# Initializing API
@app.before_first_request
def api_init():
    """
    Default API initialization
    """
    pass


# Creating FaceRecognition API
class FaceIdentification(Resource):

    # Initializing default self.results
    def __init__(self):
        self.frame_str = None
        self.request_id = None
        self.rotation = None
        self.results = {'error_message': [],
                        'face_is_detected': False,
                        'face_is_identified': False,
                        'face_similarity': 0,
                        'identified_id': -1,
                        'identified_name': '',
                        'times': {'Initialization': 0,
                                  'Face_detection': 0,
                                  'Face_identification': 0,
                                  'Response_preparing': 0,
                                  }
                        }

    # Checking the request format
    @staticmethod
    def is_the_request_format_correct():
        """
        This function is responsible for checking the request format

        Returns
        -------
        bool
            True if the request format is correct. False otherwise
        """

        if 'frame' not in request.form or 'request_id' not in request.form:
            return False
        else:
            return True

    # Extracting request information
    def extract_request_information(self, start_time):
        """
        This function is responsible for initializing the main information from the input request.

        Parameters
        ----------
        start_time : float
            The beginning time of the initializing
        """
        self.frame_str = base64.b64decode(request.form['frame'].encode('utf8'))
        self.request_id = utils.add_time(request_id=request.form['request_id'])
        self.rotation = request.form['rotation'] if 'rotation' in request.form else '0'
        self.results['times']['Initialization'] = round(time.time() - start_time, 2)

        logger.info(f'Identification request initialized with id: {self.request_id}')

    def post(self):

        start_time = time.time()

        # Checking the correctness of the request format
        if not self.is_the_request_format_correct():
            return utils.prepare_the_error_message(logger_model=logger,
                                                   results=self.results,
                                                   error_message=['Bad request, Follow the documentation'],
                                                   status_code=400)

        self.extract_request_information(start_time)

        # Face identification
        logger.debug(f'ID: {self.request_id} - Face identification started')
        try:
            self.results = face_recognition.face_identifier(self.frame_str, self.rotation, self.results)
        except RuntimeError:
            return utils.prepare_the_error_message(logger_model=logger,
                                                   results=self.results,
                                                   error_message=[],
                                                   status_code=501,
                                                   message_title='Error in face identification: ')

        logger.debug(f'ID: {self.request_id} - Face identification ended', )

        # Preparing the response
        start_time = time.time()

        logger.debug(f'ID: {self.request_id} - Start Preparing the response')
        try:
            output_error_message = [utils.remove_internal_errors(error_line=error_line) for error_line in
                                    self.results['error_message']]
            output_dict = {
                'face_is_detected': int(self.results['face_is_detected']),
                'face_is_identified': int(self.results['face_is_identified']),
                'face_similarity': int(self.results['face_similarity']),
                'identified_id': int(self.results['identified_id']),
                'identified_name': self.results['identified_name'],
                'error_message': '\n'.join(output_error_message),
            }

            response = jsonify(output_dict)

        except:
            raise RuntimeError(f'Output cannot be generated: {sys.exc_info()[0]}: {sys.exc_info()[1]}')

        self.results['times']['Response_preparing'] = round(time.time() - start_time, 1)

        # Storing the output response in logging file
        try:
            log_message = 'request_id: {}, ' \
                          'face_is_detected: {}, ' \
                          'face_is_identified: {}, ' \
                          'face_similarity: {}, ' \
                          'identified_id: {}, ' \
                          'identified_name: {}, ' \
                          'times: {}'.format(self.request_id,
                                             int(self.results['face_is_detected']),
                                             int(self.results['face_is_identified']),
                                             int(self.results['face_similarity']),
                                             int(self.results['identified_id']),
                                             self.results['identified_name'],
                                             self.results['times'])

            if len(self.results['error_message']):
                logger.error(', '.join(self.results['error_message']))

            logger.info(log_message)

        except:
            raise RuntimeError(f'Internal Logging failed: {sys.exc_info()[0]}: {sys.exc_info()[1]}')

        return response


class FaceEnrollment(Resource):

    # Initializing default self.results
    def __init__(self):
        self.frame_str = None
        self.user_id = None
        self.user_name = None
        self.request_id = None
        self.rotation = None
        self.results = {'error_message': [],
                        'face_is_detected': False,
                        'face_is_identified': False,
                        'face_similarity': 0,
                        'identified_id': -1,
                        'identified_name': '',
                        'id_is_duplicated': False,
                        'times': {'Initialization': 0,
                                  'Face_detection': 0,
                                  'Face_identification': 0,
                                  'Face_enrollment': 0,
                                  'Response_preparing': 0,
                                  }
                        }

    # Checking the request format
    @staticmethod
    def is_the_request_format_correct():
        """
        This function is responsible for checking the request format

        Returns
        -------
        bool
            True if the request format is correct. False otherwise
        """

        if 'frame' not in request.form or \
                'user_id' not in request.form or \
                'user_name' not in request.form or \
                'request_id' not in request.form:
            return False
        # TODO check if user_id is more than 0
        # TODO check if user_name length is more than 0
        else:
            return True

    # Extracting request information
    def extract_request_information(self, start_time):
        """
        This function is responsible for initializing the main information from the input request.

        Parameters
        ----------
        start_time : float
            The beginning time of the initializing
        """
        self.frame_str = base64.b64decode(request.form['frame'].encode('utf8'))
        self.user_id = int(request.form['user_id'])
        self.user_name = request.form['user_name']
        self.request_id = utils.add_time(request_id=request.form['request_id'])
        self.rotation = request.form['rotation'] if 'rotation' in request.form else '0'
        self.results['times']['Initialization'] = round(time.time() - start_time, 1)

        logger.info(f'Enrollment request initialized with id: {self.request_id}')

    def post(self):

        start_time = time.time()

        # Checking the correctness of the request format
        if not self.is_the_request_format_correct():
            return utils.prepare_the_error_message(logger_model=logger,
                                                   results=self.results,
                                                   error_message=['Bad request, Follow the documentation'],
                                                   status_code=400)

        self.extract_request_information(start_time)

        #
        logger.debug(f'ID: {self.request_id} - Face enrollment started')
        try:
            self.results = face_recognition.face_enroller(self.frame_str, self.user_id, self.user_name,
                                                          self.rotation, self.results)
            if self.results['id_is_duplicated']:
                if self.results['identified_id'] > 0:
                    return utils.prepare_the_error_message(
                        logger_model=logger,
                        results=self.results,
                        error_message=[],
                        status_code=401,
                        message_title='Error in face enrollment: ')
                elif len(self.results['identified_name']) > 0:
                    return utils.prepare_the_error_message(
                        logger_model=logger,
                        results=self.results,
                        error_message=[],
                        status_code=402,
                        message_title='Error in face enrollment: ')
                if self.results['face_is_identified']:
                    return utils.prepare_the_error_message(
                        logger_model=logger,
                        results=self.results,
                        error_message=[],
                        status_code=403,
                        message_title='Error in face enrollment: ')
        except RuntimeError:
            return utils.prepare_the_error_message(logger_model=logger,
                                                   results=self.results,
                                                   error_message=[],
                                                   status_code=502,
                                                   message_title='Error in face enrollment: ')

        logger.debug(f'ID: {self.request_id} - Face enrollment ended', )

        # Preparing the response
        start_time = time.time()

        logger.debug(f'ID: {self.request_id} - Start Preparing the response')
        try:
            output_error_message = [utils.remove_internal_errors(error_line=error_line) for error_line in
                                    self.results['error_message']]
            output_dict = {
                'face_is_detected': int(self.results['face_is_detected']),
                'face_is_identified': int(self.results['face_is_identified']),
                'face_similarity': int(self.results['face_similarity']),
                'identified_id': int(self.results['identified_id']),
                'identified_name': self.results['identified_name'],
                'error_message': '\n'.join(output_error_message),
            }

            response = jsonify(output_dict)

        except:
            raise RuntimeError(f'Output cannot be generated: {sys.exc_info()[0]}: {sys.exc_info()[1]}')

        self.results['times']['Response_preparing'] = round(time.time() - start_time, 1)

        # Storing the output response in logging file
        try:
            log_message = 'request_id: {}, ' \
                          'face_is_detected: {}, ' \
                          'face_is_identified: {}, ' \
                          'face_similarity: {}, ' \
                          'identified_id: {}, ' \
                          'identified_name: {}, ' \
                          'times: {}'.format(self.request_id,
                                             int(self.results['face_is_detected']),
                                             int(self.results['face_is_identified']),
                                             int(self.results['face_similarity']),
                                             int(self.results['identified_id']),
                                             self.results['identified_name'],
                                             self.results['times'])

            if len(self.results['error_message']):
                logger.error(', '.join(self.results['error_message']))

            logger.info(log_message)

        except:
            raise RuntimeError(f'Internal Logging failed: {sys.exc_info()[0]}: {sys.exc_info()[1]}')

        return response


# Creating HealthCheck API for checking the availability of the service
class HealthCheck(Resource):
    """
    This API is responsible for checking the face recognition service.
    It returns green status if the service is up.
    """

    @staticmethod
    def get():
        """
        Return the status

        Returns
        -------
        str
            A dictionary contains the status
        """

        output_dict = {"status": "Green"}
        response = jsonify(output_dict)

        return response


# Adding the resources to the Face Identification API
api.add_resource(FaceIdentification, '/face_recognition/face_recognition')
api.add_resource(FaceEnrollment, '/face_recognition/face_enrollment')
api.add_resource(HealthCheck, '/face_recognition/health_check')

# Starting the API from here
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Identification API.')
    parser.add_argument('--host',
                        type=str,
                        default='127.0.0.1',
                        help='End point of the server (default: 127.0.0.1)')
    parser.add_argument('--port',
                        type=str,
                        default='6000',
                        help='PORT (default: 6000)')

    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
