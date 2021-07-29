class Frame:
    def __init__(self, frame_id=None, frame_array=None, faces_info=None):
        self.frame_id = frame_id
        self.frame_array = frame_array
        self.faces_info = faces_info

    def get_frame_id(self):
        return self.frame_id

    def set_frame_id(self, value):
        self.frame_id = value

    def get_frame_array(self):
        return self.frame_array

    def set_frame_array(self, value):
        self.frame_array = value

    def get_faces_info(self):
        return self.faces_info

    def set_faces_info(self, value):
        self.faces_info = value
        
    def get_all_face_boxes(self):
        return list(map(lambda x: x.get_bounding_box(),self.get_faces_info()['facial-features']))

    def get_all_face_embeddings(self):
        return list(map(lambda x: x.get_face_encoding(), self.get_faces_info()["facial-features"]))

    def get_all_face_gender(self):
        return list(map(lambda x: x.get_gender(), self.get_faces_info()["facial-features"]))

    def get_all_face_age(self):
        return list(map(lambda x: x.get_age(), self.get_faces_info()["facial-features"]))

    def get_all_face_emotion(self):
        return list(map(lambda x: x.get_emotion(), self.get_faces_info()["facial-features"]))

