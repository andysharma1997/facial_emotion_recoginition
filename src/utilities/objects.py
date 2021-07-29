import os

from src.utilities import sken_logger
from src.services import frame_processing, face_proximity_service
import keras
import cv2
import imutils
from concurrent.futures import ThreadPoolExecutor
from mtcnn.mtcnn import MTCNN
import numpy as np

logger = sken_logger.get_logger('objects')



class Frame:
    def __init__(self, frame_number):
        self.frame_number = frame_number
        self.frame_array = None
        self.face_info = []

    def insert_face_count(self, face_count):
        self.face_count = face_count

    def insert_frame_array(self, frame_array):
        self.frame_array = frame_array

    def insert_face_info(self, face_boxes, facial_features):
        for box, feature in zip(face_boxes, facial_features):
            self.face_info.append(
                {"face_box": box,
                 "emotion": feature['emotion_output'],
                 "age": feature['age_output'],
                 "sex": feature['gender_output']})

    def get_face_info(self, index=None):
        if index is not None:
            return self.face_info[index]
        return self.face_info

    def get_all_face_positions(self):
        if len(self.face_info) > 0:
            face_boxes = []
            for face in self.face_info:
                x, y, w, h = face['face_box']
                face_boxes.append([x, y])
            return face_boxes
        else:
            return []


class Video:
    def __init__(self, video_name, video_path, total_frames, fps, skipping_frames):
        self.name = video_name
        self.path = video_path
        self.total_frames = total_frames
        self.fps = fps
        self.skipping_frames = skipping_frames
        self.batch_size = 5  # int(int(total_frames / fps) / skipping_frames)
        self.frame_computer = []
        self.all_frames = []

    def put_frame(self, frame: Frame):
        self.frame_computer.append(frame)
        logger.info("Inserting frame_id={} {}/{}".format(frame.frame_number, len(self.frame_computer), self.batch_size))
        if len(self.frame_computer) >= self.batch_size:
            self.all_frames.extend(self.frame_computer)
            f1 = self.frame_computer[0]
            for i in range(1, self.batch_size - 1):
                f2 = self.frame_computer[i + 1]
                frame_proximity_results = face_proximity_service.face_proximity_detection(f1, f2, 0.1)
                with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as exe:
                    face_box_result = exe.submit(face_proximity_service.get_frame_box_combinations, f1, f2,
                                                 frame_proximity_results).result()
                    age_result = exe.submit(face_proximity_service.get_frame_features_combinations, f1, f2,
                                            frame_proximity_results, 'age').result()
                    sex_result = exe.submit(face_proximity_service.get_frame_features_combinations, f1, f2,
                                            frame_proximity_results, 'sex').result()
                    emotion_result = exe.submit(face_proximity_service.get_frame_features_combinations, f1, f2,
                                                frame_proximity_results, 'emotion').result()

                feature_output = []
                for sex, age, emotion in zip(sex_result, age_result, emotion_result):
                    feature_output.append({"gender_output": sex, "age_output": age, "emotion_output": emotion})
                frame_number = str(f1.frame_number) + "_" + str(f2.frame_number)
                f1 = Frame(frame_number)
                f1.insert_frame_array(f2.frame_array)
                f1.insert_face_info(face_box_result, feature_output)
            self.frame_computer = []
            logger.info("Processed {} frames for video")
            return f1

        else:
            return None

    def get_all_frames(self):
        return self.all_frames


class FerModel:
    __instance__ = None
    fer_model = None
    detection_model = None
    dataset_dict = None

    @staticmethod
    def get_instance():
        if FerModel.__instance__ is None:
            logger.info("Calling private constructor for model initialization")
            FerModel()
        return FerModel.__instance__

    def __init__(self, fer_model_path, detection_model_path=None):
        if FerModel.__instance__ is not None:
            raise Exception("The singleton is already initialized you are attempting to initialize it again get lost")
        else:
            logger.info("Initializing Models")
            self.fer_model = keras.models.load_model(fer_model_path)

            self.detection_model = MTCNN()  # cv2.CascadeClassifier(detection_model_path)
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

            FerModel.__instance__ = self

    def __draw(self, frame, bounding_box_coordinates=None, fer_values=None):
        try:
            if bounding_box_coordinates is None or fer_values == None:
                cv2.imshow('output', frame)
            else:
                for (x, y, w, h), (gender, age, emotion) in zip(bounding_box_coordinates, fer_values):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, f'age:{age}|:sex{gender}|emotion:{emotion}', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255), 2)
                cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
                # cv2.putText(frame, f'Total Persons : {persons - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
                cv2.imshow('output', frame)
        except Exception as e:
            pass

    def classify_output(self, fer_predictions):
        gender, age, emotion = fer_predictions
        gender, age, emotion = gender[0], age[0], emotion[0]
        gender_idx, age_idx, emotion_idx = np.argmax(gender), np.argmax(age), np.argmax(emotion)
        return {
            "gender_output": {self.dataset_dict['gender_id'][gender_idx]: format(gender[gender_idx], '.2f')},
            "age_output": {self.dataset_dict["age_id"][age_idx]: format(age[age_idx], '.2f')},
            "emotion_output": {self.dataset_dict['emotion_id'][emotion_idx]: format(emotion[emotion_idx], '.2f')}
        }

    def predict_fer(self, image_array):
        fer_output = self.fer_model.predict_generator(image_array)
        return self.classify_output(fer_output)

    def run(self, filepath, frames_to_skip=5):
        logger.info("Reading file {} with frame skipping={}".format(filepath, frames_to_skip))
        cap = cv2.VideoCapture(filepath)
        file_dir, file_name = os.path.split(filepath)
        total_video_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        video = Video(file_name, file_dir, total_video_frames, fps, frames_to_skip)
        f_count = 1
        batch_frame_state = None
        while True:
            ret, frame = cap.read()
            if ret:
                if f_count % frames_to_skip == 0:
                    frame = imutils.resize(frame, width=min(800, frame.shape[1]))
                    # boxes = self.detection_model.detectMultiScale(frame, scaleFactor=1.1)
                    resp = self.detection_model.detect_faces(frame)
                    boxes = [item['box'] for item in resp]
                    if len(boxes) > 0:
                        frame_obj = Frame(f_count)
                        frame_obj.insert_face_count(len(boxes))
                        frame_obj.insert_frame_array(frame)
                        with ThreadPoolExecutor(max_workers=100) as exe:
                            face_frames = list(
                                exe.map(frame_processing.crop_face_from_frame, [frame] * len(boxes), boxes))
                        with ThreadPoolExecutor(max_workers=100) as exe:
                            processed_face_frames = list(exe.map(frame_processing.process_image, face_frames))
                        fer_values = []
                        for f in processed_face_frames:
                            fer_values.append(self.predict_fer(np.array([f])))
                        frame_obj.insert_face_info(boxes, fer_values)
                        video_insert_resp = video.put_frame(frame_obj)
                        if video_insert_resp is not None:
                            batch_frame_state = video_insert_resp
                            boxes = []
                            fer_values = []
                            for face in video_insert_resp.get_face_info():
                                boxes.append(face['face_box'])
                                fer_values.append([face['sex'], face['age'], face['emotion']])
                            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                            print(boxes)
                            print(fer_values)
                            self.__draw(video_insert_resp.frame_array, fer_values)
                        else:
                            if batch_frame_state == None:
                                batch_frame_state = frame_obj
                            print("********************************************")
                            print(batch_frame_state.get_all_face_positions(), frame_obj.get_all_face_positions())
                            proxy_result = face_proximity_service.face_proximity_detection(batch_frame_state, frame_obj,
                                                                                           0.1)
                            print(proxy_result)
                            boxes = []
                            fer_val = []
                            for key, val in proxy_result.items():
                                if val is not None:
                                    boxes.append(frame_obj.get_face_info()[val]['face_box'])
                                    fer_val.append([batch_frame_state.get_face_info()[key]['sex'],
                                                    batch_frame_state.get_face_info()[key]['age'],
                                                    batch_frame_state.get_face_info()[key]['emotion']])
                                else:
                                    boxes.append(batch_frame_state.get_face_info([key]['face_box']))
                                    fer_val.append([batch_frame_state.get_face_info()[key]['sex'],
                                                    batch_frame_state.get_face_info()[key]['age'],
                                                    batch_frame_state.get_face_info()[key]['emotion']])
                            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                            print(boxes)
                            print(fer_val)
                            self.__draw(frame_obj.frame_array, boxes, fer_val)
                    else:
                        fer_values = boxes = None
                        self.__draw(frame, boxes, fer_values)
            else:
                break
            logger.info("Done Frame {}/{}".format(f_count, total_video_frames))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            f_count += 1
        cap.release()
        cv2.destroyAllWindows()
        logger.info('Released the video')


if __name__ == '__main__':
    # detection_model_path = '/home/andy/Desktop/zoom/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
    # fer_model_path = "/home/andy/Desktop/sken_project/facial_emotion_detection/model/trained_andy"
    # file_path = "/home/andy/Downloads/7_tEmd7Zuw_EKqK.mp4"
    # FerModel(fer_model_path=fer_model_path, detection_model_path=detection_model_path)
    # FerModel.get_instance().run(file_path)
    f_obj = FacialFeature
    f_obj.face_encoding
    f_obj.face_encoding.fget()
