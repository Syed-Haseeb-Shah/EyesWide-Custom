import cv2 as cv

class Camera:

    def __init__(self):
        self.camera = cv.VideoCapture(0)
        if not self.camera.isOpened():
            raise ValueError("Unable to open camera!")

        self.width = int(self.camera.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv.CAP_PROP_FRAME_HEIGHT))

    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()

    def get_frame(self):
        if self.camera.isOpened():
            ret, frame = self.camera.read()

            if ret:
                return ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return None

if __name__ == "__main__":
    cam = Camera()
    while True:
        ret, frame = cam.get_frame()
        if not ret:
            print("Error reading frame from camera.")
            break

        cv.imshow("Camera Stream", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
