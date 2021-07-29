import cv2
from src.utilities import sken_logger, frame_class
import os
from scipy.stats import mode as mp
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = sken_logger.get_logger('video_class')


class Video:
    def __init__(self, video_name, video_path, skipping_frames, web_cam=False, total_frames=None, fps=None):
        self.name = video_name
        self.path = video_path
        self.total_frames = total_frames
        self.fps = fps
        self.skipping_frames = skipping_frames
        self.all_frames = []
        self.web_cam = web_cam

    def get_name(self):
        return self.name

    def set_name(self, value):
        self.name = value

    def get_path(self):
        return self.path

    def set_path(self, value):
        self.path = value

    def get_total_frames(self):
        return self.total_frames

    def set_total_frames(self, value):
        self.total_frames = value

    def get_fps(self):
        return self.fps

    def set_fps(self, value):
        self.fps = value

    def get_skipping_frames(self):
        return self.skipping_frames

    def set_skipping_frames(self, value):
        self.skipping_frames = value

    def get_all_frame(self):
        return self.all_frames

    def insert_frame(self, value: frame_class.Frame):
        self.all_frames.append(value)

    def generate_frames(self):
        logger.info("Generating frames for video_file={} at path={}".format(self.name, self.path))
        file_path = os.path.join(self.path, self.name)
        if self.web_cam:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(file_path)
            assert os.path.exists(file_path), 'File {} could not be found'.format(file_path)
        self.set_total_frames(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.set_fps(cap.get(cv2.CAP_PROP_FPS))
        frame_number = 1
        while True:
            ret, frame = cap.read()
            if ret:
                if frame_number % self.get_skipping_frames() == 0:
                    frame_obj = frame_class.Frame(frame_id=frame_number,
                                                  frame_array=frame)
                    yield frame_obj
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_number += 1
            print("Done frame = {}/{}".format(frame_number, self.get_total_frames()))
        logger.info("Releasing the video file = {}".format(file_path))
        cap.release()

    # def draw_output(self):
    #     logger.info("Starting Drawing the file {}".format(os.path.join(self.path, self.name)))
    #     for i in range(2, len(self.all_frames)):
    #         f1, f2, f3 = self.all_frames[i], self.all_frames[i - 1], self.all_frames[i - 2]
    #         if f1.get_faces_info()['face_counts'] is None:
    #             continue
    #         f1_face_embeddings, f2_face_embeddings, f3_face_embeddings = f1.get_all_face_embeddings(), f2.get_all_face_embeddings(), f3.get_all_face_embeddings()
    #         cos_sim12, cos_sim23 = cosine_similarity(f1_face_embeddings, f2_face_embeddings), cosine_similarity(
    #             f2_face_embeddings, f3_face_embeddings)
    #         emotion = []
    #         age = []
    #         gender = []
    #         for j, row in enumerate(np.argmax(cos_sim12, axis=1)):
    #             if cos_sim12[j][row] >= 0.9:
    #                 delta_emotion, delta_age, delta_gender = [f1.get_all_face_emotion()[j],
    #                                                           f2.get_all_face_emotion()[row]], [
    #                                                              f1.get_all_face_age()[j],
    #                                                              f2.get_all_face_age()[row]], [
    #                                                              f1.get_all_face_gender()[j],
    #                                                              f2.get_all_face_gender()[row]]
    #                 max_2 = np.argmax(cos_sim23[row])
    #                 if cos_sim23[row][max_2] > 0.9:
    #                     delta_emotion.append(f3.get_all_face_emotion()[max_2])
    #                     delta_age.append(f3.get_all_face_age()[max_2])
    #                     delta_gender.append(f3.get_all_face_gender()[max_2])
    #                 emotion.append(delta_emotion)
    #                 age.append(delta_age)
    #                 gender.append(delta_gender)
    #         frame = f1.get_frame_array()
    #         for (x, y, x_, y_), gen, umar, emo in zip(f1.get_all_face_boxes(), gender, age, emotion):
    #             cv2.rectangle(frame, (int(x), int(y)), (int(x_), int(y_)), (0, 0, 255), 2)
    #             print("$$$$$$$$$$$$$$$$$$$$$$$$")
    #             print(emo)
    #             cv2.putText(frame, f'emotion:{mp(emo).mode[0]}|age:{mp(umar).mode[0]}|sex:{mp(gen).mode[0]}',
    #                         (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
    #                         0.5,
    #                         (0, 0, 255), 2)
    #         cv2.imshow('output', frame)
    #         cv2.waitKey(500)

    def draw_output(self, frame_queue):
        logger.info("Starting Drawing the file {}".format(os.path.join(self.path, self.name)))
        if frame_queue == None:
            return
        f1, f2, f3 = frame_queue
        if f1.get_faces_info()['face_counts'] is None:
            return
        f1_face_embeddings, f2_face_embeddings, f3_face_embeddings = f1.get_all_face_embeddings(), f2.get_all_face_embeddings(), f3.get_all_face_embeddings()
        cos_sim12, cos_sim23 = cosine_similarity(f1_face_embeddings, f2_face_embeddings), cosine_similarity(
            f2_face_embeddings, f3_face_embeddings)
        emotion = []
        age = []
        gender = []
        for j, row in enumerate(np.argmax(cos_sim12, axis=1)):
            if cos_sim12[j][row] >= 0.9:
                delta_emotion, delta_age, delta_gender = [f1.get_all_face_emotion()[j],
                                                          f2.get_all_face_emotion()[row]], [
                                                             f1.get_all_face_age()[j],
                                                             f2.get_all_face_age()[row]], [
                                                             f1.get_all_face_gender()[j],
                                                             f2.get_all_face_gender()[row]]
                max_2 = np.argmax(cos_sim23[row])
                if cos_sim23[row][max_2] > 0.9:
                    delta_emotion.append(f3.get_all_face_emotion()[max_2])
                    delta_age.append(f3.get_all_face_age()[max_2])
                    delta_gender.append(f3.get_all_face_gender()[max_2])
                emotion.append(delta_emotion)
                age.append(delta_age)
                gender.append(delta_gender)
        frame = f1.get_frame_array()
        for (x, y, x_, y_), gen, umar, emo in zip(f1.get_all_face_boxes(), gender, age, emotion):
            cv2.rectangle(frame, (int(x), int(y)), (int(x_), int(y_)), (0, 0, 255), 2)
            # |age:{mp(umar).mode[0]}|sex:{mp(gen).mode[0]}
            cv2.putText(frame, f'emotion:{mp(emo).mode[0]}',
                        (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255), 2)
        cv2.imshow('output', frame)
        cv2.waitKey(10)
