"""
This class contains all the requirements for face recognition tasks.
These requirements include main model and related functions.
"""
import json
import os
import sys
import time

import cv2
import insightface
import numpy as np

from config import ENROLLED_EMBEDDING_ADDRESS, MAX_IMAGE_SCALE


class FaceIdentification:
    def __init__(self):
        # Initializing face detection and face recognition models
        self.model = insightface.app.FaceAnalysis(det_name='retinaface_mnet025_v2',
                                                  rec_name='arcface_mfn_v1',
                                                  ga_name=None)

        # Change ctx-id to a positive number to use gpu for FaceAnalysis model.
        # The nms threshold is set to 0.4 by default.
        self.model.prepare(ctx_id=-1)

        # Default face verification threshold
        self.face_verification_threshold = self._score_normalizer(0.36042076)

        # Load enrolled templates
        self.enrolled_embeddings_path = ENROLLED_EMBEDDING_ADDRESS
        self._enrolled_embeddings_loader()

    @staticmethod
    def _score_normalizer(similarity_score):
        """
        Computing normalized score based on input number.
        Maps cosine similarity to a number between 0 and 100.
        The normalizer is a function to regress:
        the cumulative histogram of similarity of mates for the outputs of more than 50 and
        the negative of cumulative histogram of similarity of non-mates for the outputs of less than 50.
        Histograms are extracted from LFW dataset.

        Parameters
        ----------
        similarity_score : float
            Input similarity score

        Returns
        -------
        float
            Normalized score
        """

        y = [-0.7, -0.25, 0., 0.4, 0.75]
        x = [-0.046097733, 0.07163449, 0.36042076, 0.6658068, 0.76499796]
        coefficient = [17., 15., 0., 10., 17.]
        a_plus = 3.528539219975098
        b_plus = -1.9493253020156809
        a_minus = 3.8222332195969924
        b_minus = -0.5238037132720291

        if similarity_score < x[0]:
            normalized_score = -((1 - np.exp((similarity_score - x[0]) * coefficient[0])) * (1 + y[0])) + y[0]
        elif similarity_score < x[1]:
            normalized_score = a_minus * similarity_score + b_minus
        elif similarity_score < x[2]:
            normalized_score = (np.exp((-similarity_score + x[1]) * coefficient[1]) - 1) * (y[1] - 0) + y[1]
        elif similarity_score < x[3]:
            normalized_score = (np.exp((similarity_score - x[3]) * coefficient[3]) - 1) * (y[3] - 0) + y[3]
        elif similarity_score < x[4]:
            normalized_score = a_plus * similarity_score + b_plus
        else:
            normalized_score = (1 - np.exp(-(similarity_score - x[4]) * coefficient[4])) * (1 - y[4]) + y[4]

        if normalized_score > 0.995:
            normalized_score = 1

        return (normalized_score + 1.) * 50.

    # Load enrolled embeddings
    def _enrolled_embeddings_loader(self):
        """
        Loads enrolled embeddings from a json file.
        """

        enrolled_embeddings_file = os.path.join(self.enrolled_embeddings_path, 'enrolled_embeddings.json')
        if os.path.exists(enrolled_embeddings_file):
            with open(enrolled_embeddings_file, 'r') as f:
                self.enrolled_embeddings = json.load(f)
        else:
            # TODO make directory & an empty json file
            self.enrolled_embeddings = {}
        return

    def _enrolled_embeddings_adder(self, embedding, user_id, user_name):
        """
        Adds an embedding to the enrolled_embeddings variable.

        Parameters
        ----------
        embedding : list
            The normed feature vector extracted by identification model from the face of the new user.
        user_id : int
            The id of the new user.
        user_name : str
            The name of the new user

        """

        self.enrolled_embeddings[user_id] = {'name': user_name, 'embedding': embedding}
        return

    def _enrolled_embeddings_saver(self):
        """
        Save enrolled embeddings to the json file.
        """

        with open(os.path.join(self.enrolled_embeddings_path, 'enrolled_embeddings.json'), 'w') as f:
            json.dump(self.enrolled_embeddings, f)
        return

    # Extracting face features
    def _face_feature_extractor(self, image):
        """
        Extracting a face from an input image using predefined model.

        Parameters
        ----------
        image : object
            An array of input image as a ndarray

        Return
        ------
        object
            Detected face landmark as a numpy array
        """

        try:
            faces = self.model.get(image)
        except:
            raise RuntimeError(f'InsightFace error : {sys.exc_info()[0]}: {sys.exc_info()[1]}')

        if len(faces) == 0:
            raise RuntimeError('No face found in the image')

        if len(faces) > 1:
            raise RuntimeError('More than one face found in the image')

        return faces[0]

    # Downscaling the input image
    @staticmethod
    def _image_scaler(img, max_scale):
        """
        Downscale input image into max scale size.

        Parameters
        ----------
        img : object
            An array of input image as ndarray
        max_scale : float
            Maximum scale size

        Returns
        -------
        img : object
            An array of scaled image

        scale : float
            Scale factor
        """

        scale = 1.

        if max(img.shape) > max_scale:
            scale = max_scale / max(img.shape)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        return img, scale

    # Converting an image
    @staticmethod
    def _image_format_convertor(img_str):
        """
        Convert image string into cv2 image

        Parameters
        ----------
        img_str : str
            Input image as bytes stream

        Returns
        -------
        object
            Converted image string into cv2 image as ndarray
        """

        np_img = np.fromstring(img_str, np.uint8)
        gallery_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        assert gallery_img is not None

        return gallery_img

    # Computing face similarity
    def face_identifier(self, frame_str, rotation, results):
        """
        Detect and identify the face in the frame from enrolled faces.

        Parameters
        ----------
        frame_str : str
            Input frame as bytes stream
        rotation : int
            rotation of frames (0: no rotation, 1: 90 degrees, 2: 180 degrees, 3: 270 degrees)
        results : dict
            A dictionary of results

        Returns
        -------
        results : dict
            A dictionary of results
        """

        # Decoding string image into cv2 image
        try:
            frame = self._image_format_convertor(frame_str)
        except:
            raise RuntimeError(f'Image frame cannot be loaded: {sys.exc_info()[0]}: {sys.exc_info()[1]}')

        # Downscaling cv2 image to default max image scale
        try:
            frame, _ = self._image_scaler(frame, MAX_IMAGE_SCALE)
        except:
            raise RuntimeError(f'Image frame cannot be down scaled: {sys.exc_info()[0]}: {sys.exc_info()[1]}')

        # TODO rotate face

        # Extracting face from an image
        try:
            start_time = time.time()
            probe_face = self._face_feature_extractor(frame)
            probe_normed_embedding = probe_face.normed_embedding
            results['face_is_detected'] = True
            results['times']['Face_detection'] = round(time.time() - start_time, 2)
        except:
            raise RuntimeError(f'Embedding cannot be extracted from Image: {sys.exc_info()[0]}: {sys.exc_info()[1]}')

        # Computing face similarities between this frame's embedding and enrolled embeddings
        try:
            start_time = time.time()
            face_similarities = []
            enrolled_ids = list(self.enrolled_embeddings.keys())
            for enrolled_id in enrolled_ids:
                sim = int(self._score_normalizer(
                    np.dot(probe_normed_embedding, self.enrolled_embeddings[enrolled_id]['embedding']).item()
                ))
                face_similarities.append(sim)
            face_similarities = np.array(face_similarities)
            results['times']['Face_identification'] = round(time.time() - start_time, 2)
        except:
            raise RuntimeError(f'Face Similarities cannot be extracted from frame: '
                               f'{sys.exc_info()[0]}: {sys.exc_info()[1]}')

        # Find identified id and name
        try:
            identified_ind = np.where(face_similarities > self.face_verification_threshold)[0].tolist()
            if len(identified_ind) > 1:
                identified_ids = [list(self.enrolled_embeddings.keys())[ind] for ind in identified_ind]
                results['error_message'].append(
                    f'Warning! Face is matched with more than one id. Matched ids: {identified_ids}')
                identified_ind = [np.argmax(face_similarities)]

            if len(identified_ind) == 1:
                results['face_is_identified'] = True
                results['face_similarity'] = face_similarities[identified_ind]
                results['identified_id'] = int(list(self.enrolled_embeddings.keys())[identified_ind[0]])
                results['identified_name'] = list(self.enrolled_embeddings.values())[identified_ind[0]]['name']
        except:
            raise RuntimeError(f'Error in finding identified id and name: '
                               f'{sys.exc_info()[0]}: {sys.exc_info()[1]}')
        return results

    def face_enroller(self, frame_str, user_id, user_name, rotation, results):
        """
        Identify detected face in the frame from enrolled faces.

        Parameters
        ----------
        frame_str : str
            Input frame as bytes stream
        user_id : int
            The id of the new user.
        user_name : str
            The name of the new user
        rotation : int
            rotation of frames (0: no rotation, 1: 90 degrees, 2: 180 degrees, 3: 270 degrees)
        results : dict
            A dictionary of results

        Returns
        -------
        results : dict
            A dictionary of results
        """

        # Return if user_id already exists in enrolled_embeddings
        if str(user_id) in self.enrolled_embeddings.keys():
            results['id_is_duplicated'] = True
            results['identified_id'] = user_id

            identified_name = self.enrolled_embeddings[str(user_id)]['name']
            results['error_message'].append(f'The id {user_id} already exists with name: {identified_name}')
            return results
        # Return if user_name already exists in enrolled_embeddings
        user_names = [value['name'] for value in self.enrolled_embeddings.values()]
        if user_name in user_names:
            results['id_is_duplicated'] = True
            results['identified_name'] = user_name

            identified_id = list(self.enrolled_embeddings.keys())[user_names.index(user_name)]
            results['error_message'].append(f'The name {user_name} already exists with id: {identified_id}')
            return results

        # Decoding string image into cv2 image
        try:
            frame = self._image_format_convertor(frame_str)
        except:
            raise RuntimeError(f'Image frame cannot be loaded: {sys.exc_info()[0]}: {sys.exc_info()[1]}')

        # Downscaling cv2 image to default max image scale
        try:
            frame, _ = self._image_scaler(frame, MAX_IMAGE_SCALE)
        except:
            raise RuntimeError(f'Image frame cannot be down scaled: {sys.exc_info()[0]}: {sys.exc_info()[1]}')

        # TODO rotate

        # Extracting face from an image
        try:
            probe_face = self._face_feature_extractor(frame)
            probe_normed_embedding = probe_face.normed_embedding.tolist()
        except:
            raise RuntimeError(f'Embedding cannot be extracted from Image: {sys.exc_info()[0]}: {sys.exc_info()[1]}')

        # Computing face similarities between this frame's embedding and enrolled embeddings
        try:
            face_similarities = []
            enrolled_ids = list(self.enrolled_embeddings.keys())
            for enrolled_id in enrolled_ids:
                sim = int(self._score_normalizer(
                    np.dot(probe_normed_embedding, self.enrolled_embeddings[enrolled_id]['embedding']).item()
                ))
                face_similarities.append(sim)
            face_similarities = np.array(face_similarities)
        except:
            raise RuntimeError(f'Face Similarities cannot be extracted from frame: '
                               f'{sys.exc_info()[0]}: {sys.exc_info()[1]}')

        # Check if this frame's face is already enrolled
        try:
            identified_ind = np.where(face_similarities > self.face_verification_threshold)[0].tolist()
            if len(identified_ind) > 1:
                identified_ids = [list(self.enrolled_embeddings.keys())[ind] for ind in identified_ind]
                results['error_message'].append(
                    f'Warning! Face is matched with more than one id. Matched ids: {identified_ids}')
                identified_ind = [np.argmax(face_similarities)]

            if len(identified_ind) == 1:
                results['id_is_duplicated'] = True
                results['face_is_identified'] = True
                results['face_similarity'] = face_similarities[identified_ind].item()
                results['identified_id'] = int(list(self.enrolled_embeddings.keys())[identified_ind[0]])
                results['identified_name'] = list(self.enrolled_embeddings.values())[identified_ind[0]]['name']

                identified_id = results['identified_id']
                identified_name = results['identified_name']
                results['error_message'].append(
                    f'This face has matched with id: {identified_id}, name: {identified_name}, '
                    f'and similarity: {face_similarities[identified_ind].item()}')
                return results
        except:
            raise RuntimeError(f'Error in checking if face is already enrolled: '
                               f'{sys.exc_info()[0]}: {sys.exc_info()[1]}')
        self._enrolled_embeddings_adder(probe_normed_embedding, user_id, user_name)
        self._enrolled_embeddings_saver()

        return results
