import sys, numpy, os

vision_enabled = False
try:
    import cv2
    vision_enabled = True
except Exception as e:
    print("Warning: OpenCV not installed. To use facial recognition, make sure you've properly configured OpenCV.")


class Vision(object):
    def __init__(self, facial_recognition_model="models/facial_recognition_model.xml", camera=0):
        self.facial_recognition_model = facial_recognition_model
        self.camera = camera
        self.person = "Unknown"
        (self._im_width, self._im_height) = (112, 92)
        self.train()

    def train(self):
        fn_dir = 'faces'

        # Part 1: Create Recognizer
        print('Training...')
        # Create a list of images and a list of corresponding names
        (images, labels, self._names, id) = ([], [], {}, 0)
        for (subdirs, dirs, files) in os.walk(fn_dir):
            for subdir in dirs:
                print "Training %s" %subdir
                self._names[id] = subdir
                subjectpath = os.path.join(fn_dir, subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + '/' + filename
                    label = id
                    images.append(cv2.imread(path, 0))
                    labels.append(int(label))
                id += 1

        # Create a Numpy array from the two lists above
        (images, labels) = [numpy.array(lis) for lis in [images, labels]]

        # OpenCV trains a model from the images
        # Fisher wasnt as good at recognozing faces as LBPHFace
        #model = cv2.createFisherFaceRecognizer()
        self._model = cv2.face.createLBPHFaceRecognizer()
        #model = cv2.createEigenFaceRecognizer()
        self._model.train(images, labels)

        return

    def recognize_person(self):
        """
        Wait until a face is recognized. If openCV is configured, always return true
        :return:
        """

        if vision_enabled is False:  # if opencv is not able to be imported, always return True
            return True

        face_cascade = cv2.CascadeClassifier(self.facial_recognition_model)
        video_capture = cv2.VideoCapture(self.camera)

        size = 4
        fn_haar = 'models/haarcascade_frontalface_default.xml'

        # Part 2: Use Recognizer on camera stream
        haar_cascade = cv2.CascadeClassifier(fn_haar)
        webcam = cv2.VideoCapture(0)
        while True:
            (rval, frame) = webcam.read()
            frame=cv2.flip(frame,1,0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mini = cv2.resize(gray, (gray.shape[1] / size, gray.shape[0] / size))
            faces = haar_cascade.detectMultiScale(mini)
            for i in range(len(faces)):
                face_i = faces[i]
                (x, y, w, h) = [v * size for v in face_i]
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (self._im_width, self._im_height))

                # Try to recognize the face
                prediction = self._model.predict(face_resize)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # Write the name of recognized face
                # [1]
                if prediction[1]<=100:
                    cv2.putText(frame,
                        '%s - %.0f' % (self._names[prediction[0]],prediction[1]),
                        (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                    print '%s - %.0f' % (self._names[prediction[0]],prediction[1])
                    self.person = self._names[prediction[0]]
                    return True
                else:
                    cv2.putText(frame,
                    'Unknown -  %.0f' % (prediction[1]),
                    (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                    print 'Unknown -  %.0f' % (prediction[1])
                    self.person = "Unkown"
                    return False

    def recognize_face(self):
        """
        Wait until a face is recognized. If openCV is configured, always return true
        :return:
        """

        if vision_enabled is False:  # if opencv is not able to be imported, always return True
            return True

        face_cascade = cv2.CascadeClassifier(self.facial_recognition_model)
        video_capture = cv2.VideoCapture(self.camera)

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) > 0:
                # When everything is done, release the capture
                video_capture.release()
                cv2.destroyAllWindows()

                return True


if __name__ == "__main__":
    faceCascade = cv2.CascadeClassifier("models/facial_recognition_model.xml")

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()