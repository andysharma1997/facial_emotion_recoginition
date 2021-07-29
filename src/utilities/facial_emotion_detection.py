from src.utilities import sken_logger
from keras.models import load_model
from src.utilities import constants, video_class, facial_feature_class
from facenet_pytorch import MTCNN
# from mtcnn.mtcnn import MTCNN
from src.services import frame_processing
from keras.utils import CustomObjectScope
import tensorflow as tf
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import time
import torch
import uuid
import cv2
from collections import deque

logger = sken_logger.get_logger('facial_emotion_detection')


class FerModel:
    __instance__ = None
    fer_model = None
    detection_model = None
    embedding_model = None
    dataset_dict = None
    device = None

    @staticmethod
    def get_instance():
        if FerModel.__instance__ is None:
            logger.info("Calling private constructor for model initialization")
            FerModel()
        return FerModel.__instance__

    def __init__(self):
        if FerModel.__instance__ is not None:
            raise Exception("The singleton is already initialized you are attempting to initialize it again get lost")
        else:
            logger.info("!!!!!!!!!!!Initializing models. Please wait!!!!!!!!")
            # self.detection_model = MTCNN()
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.detection_model = MTCNN(device=self.device)
            self.fer_model = load_model(constants.fetch_constant('fer_model_path'))
            with CustomObjectScope({'tf': tf}):
                self.embedding_model = load_model(constants.fetch_constant('facenet_model_path'))
            self.embedding_model.load_weights(constants.fetch_constant('facenet_model_weights_path'))
            self.dataset_dict = {
                "age_id": {
                    0: "old",
                    1: "middle",
                    2: "young",
                    3: "child"
                },
                "gender_id": {
                    0: "male",
                    1: "female"
                },
                "emotion_id": {
                    0: "neutral",
                    1: "happiness",
                    2: "surprise",
                    3: "anger",
                    4: "sadness",
                    5: "disgust",
                    6: "fear"
                }
            }
            logger.info('All models are loaded. Ready to serve (:')
            FerModel.__instance__ = self

    def extract_faces(self, image_pixels):
        # return self.detection_model.detect_faces(image_pixels)
        return self.detection_model.detect(image_pixels)

    def detect_facial_emotion(self, faces):
        fer_outputs = self.fer_model.predict(faces)
        output = []
        for gen, age, emo in zip(fer_outputs[0], fer_outputs[1], fer_outputs[2]):
            output.append({"gender": self.dataset_dict['gender_id'][np.argmax(gen)],
                           "age": self.dataset_dict['age_id'][np.argmax(age)],
                           "emotion": self.dataset_dict['emotion_id'][np.argmax(emo)]})
        return output

    def generate_face_embeddings(self, faces):
        with ThreadPoolExecutor(max_workers=100) as exe:
            faces = list(exe.map(frame_processing.standard_scaling_image, faces))
        embeddings = self.embedding_model.predict(np.array(faces))
        return embeddings

    # def draw(self,frame,bounding_box,box_inputs):

    def process_video(self, video_path=None):
        if video_path is None:
            logger.info("Taking Feed from Web camera")
            file_name = str(uuid.uuid1().fields[0]) + '.mp3'
            base_dir = "/tmp"
            web_cam = True
        else:
            base_dir, file_name = os.path.split(video_path)
            web_cam = False
        s = time.time()
        video = video_class.Video(video_name=file_name, video_path=base_dir,
                                  skipping_frames=int(constants.fetch_constant('skip_frame')), web_cam=web_cam)
        frame_queue = deque()
        max_frame_queue_len = 3
        try:
            for frame_obj in video.generate_frames():
                frame_obj.set_frame_array(
                    frame_processing.process_image(frame_obj.get_frame_array(), x_image=800,
                                                   y_image=min(800, frame_obj.get_frame_array().shape[1]),
                                                   keep_aspect=True))
                face_extraction_output = self.extract_faces(frame_obj.get_frame_array())
                if face_extraction_output[0] is None:
                    bounding_boxes = []
                else:
                    bounding_boxes = [face for face, prob in zip(face_extraction_output[0], face_extraction_output[1])
                                      if
                                      prob > 0.9]

                if len(bounding_boxes) > 0:

                    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as exe:
                        face_arrays = list(
                            exe.map(frame_processing.crop_face_from_frame,
                                    [frame_obj.get_frame_array()] * len(bounding_boxes),
                                    bounding_boxes))

                    with ThreadPoolExecutor(max_workers=500) as exe:
                        emotion_face_arrays = list(
                            exe.map(frame_processing.process_image, face_arrays, [224] * len(face_arrays),
                                    [224] * len(face_arrays)))
                        embedding_face_arrays = list(
                            exe.map(frame_processing.process_image, face_arrays, [160] * len(face_arrays),
                                    [160] * len(face_arrays)))

                    emotion_output = self.detect_facial_emotion(np.array(emotion_face_arrays))
                    embedding_output = self.generate_face_embeddings(embedding_face_arrays)
                    facial_info = {"face_counts": len(bounding_boxes), "facial-features": []}
                    for box, emotion, embed in zip(bounding_boxes, emotion_output, embedding_output):
                        facial_info['facial-features'].append(
                            facial_feature_class.FacialFeature(bounding_box=box, face_encoding=embed,
                                                               gender=emotion['gender'], age=emotion['age'],
                                                               emotion=emotion['emotion']))

                    frame_obj.set_faces_info(facial_info)
                    video.insert_frame(frame_obj)
                    frame_queue.appendleft(frame_obj)
                    if len(frame_queue) >= max_frame_queue_len:
                        video.draw_output(frame_queue)
                        frame_queue.pop()
                else:
                    frame_obj.set_faces_info({"face_counts": None})
                    cv2.imshow('output', frame_obj.get_frame_array())
                    cv2.waitKey(500)
                    del frame_obj

        except AssertionError as exe:
            logger.error(exe)
            pass
        print("Time for video ={}".format(time.time() - s))


if __name__ == '__main__':
    FerModel().get_instance().process_video('/home/andy/Downloads/7_tEmd7Zuw_EKqK.mp4')
