class FacialFeature:
    def __init__(self, bounding_box=None, face_encoding=None, gender=None, age=None, emotion=None):
        self.bounding_box = bounding_box
        self.face_encoding = face_encoding
        self.gender = gender
        self.age = age
        self.emotion = emotion

    def get_bounding_box(self):
        return self.bounding_box

    def set_bounding_box(self, value):
        self.bounding_box = value

    def get_face_encoding(self):
        return self.face_encoding

    def set_face_encoding(self, value):
        self.face_encoding = value

    def get_gender(self):
        return self.gender

    def set_gender(self, value):
        self.gender = value

    def get_age(self):
        return self.age

    def set_age(self, value):
        self.age = value

    def get_emotion(self):
        return self.emotion

    def set_emotion(self, value):
        self.emotion = value
