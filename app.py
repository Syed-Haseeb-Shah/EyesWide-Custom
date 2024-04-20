import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2 as cv
import os
import PIL.Image, PIL.ImageTk
from model import Model
from camera import Camera

class App:

    def __init__(self, window=tk.Tk(), window_title="Camera Classifier"):
        self.window = window
        self.window.title(window_title)
        self.counters = [1]  # Initialize counters list with one element for the first class
        self.model = None  # Model will be initialized later based on the number of classes
        self.auto_predict = False
        self.camera = Camera()
        self.class_names = []  # List to store class names
        self.init_gui()
        self.delay = 15
        self.update()
        self.window.attributes("-topmost", True)
        self.window.mainloop()

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=800, height=600, bg="white")  # Adjust canvas size here
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(self.window, text="Auto Prediction", width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        # Prompt user to enter the number of classes
        num_classes = simpledialog.askinteger("Number of Classes", "Enter the number of classes:")
        self.counters = [1] * num_classes  # Initialize counters list with one element for each class
        self.model = Model(num_classes)  # Initialize the model with the number of classes

        # Prompt user to enter class names
        for i in range(num_classes):
            class_name = simpledialog.askstring(f"Classname {i+1}", f"Enter the name of class {i+1}:")
            if class_name:
                self.class_names.append(class_name)
                btn_class = tk.Button(self.window, text=class_name, width=50, command=lambda i=i: self.save_for_class(i))
                btn_class.pack(anchor=tk.CENTER, expand=True)
            else:
                messagebox.showerror("Error", "Please enter a valid class name.")
                self.window.destroy()
                return

        self.btn_train = tk.Button(self.window, text="Train Model", width=50, command=self.train_model)
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text="CLASS", font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict

    def save_for_class(self, class_num):
        if self.counters[class_num] <= 10:
            ret, frame = self.camera.get_frame()
            if not os.path.exists(str(class_num)):
                os.mkdir(str(class_num))

            cv.imwrite(f'{class_num}/frame{self.counters[class_num]}.jpg', frame)
            self.counters[class_num] += 1
        else:
            messagebox.showwarning("Warning", "You have already captured 10 images for this class.")

    def reset(self):
        for folder in map(str, range(len(self.counters))):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

        self.counters = [1] * len(self.counters)
        self.model = Model(len(self.counters))
        self.class_label.config(text="CLASS")

    def train_model(self):
        if all(counter > 1 for counter in self.counters):
            image_folder_paths = map(str, range(len(self.counters)))  # Paths to folders containing training images for each class
            self.model.train_model(image_folder_paths, self.class_names)
            messagebox.showinfo("Info", "Model trained successfully!")
        else:
            messagebox.showerror("Error", "Please capture at least one image for each class.")

    def update(self):
        if self.auto_predict:
            ret, frame = self.camera.get_frame()
            if ret:
                prediction = self.model.predict(frame)
                if prediction:
                    self.class_label.config(text=f"CLASS: {prediction}")
                else:
                    self.class_label.config(text="CLASS: None")

        ret, frame = self.camera.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

app = App()
