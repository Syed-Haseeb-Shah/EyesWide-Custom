import numpy as np
import cv2 as cv
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model as KerasModel

class Model:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        self.model = KerasModel(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False
        self.class_names = []

    def train_model(self, image_folder_paths, class_names):
        img_list = []
        class_list = []
        for class_index, folder_path in enumerate(image_folder_paths):
            for i, file_name in enumerate(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, file_name)
                img = cv.imread(file_path)
                img = cv.resize(img, (224, 224))
                img = img / 255.0
                img_list.append(img)
                class_list.append(class_index)

        img_array = np.array(img_list)
        class_array = np.array(class_list)

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(img_array, class_array, epochs=10, batch_size=32, validation_split=0.2)
        self.class_names = class_names

    def predict(self, frame):
        img = cv.resize(frame, (224, 224))
        img = img / 255.0
        prediction = self.model.predict(np.expand_dims(img, axis=0))
        predicted_class_index = np.argmax(prediction)
        if predicted_class_index < self.num_classes:
            predicted_class_name = self.class_names[predicted_class_index]
            return predicted_class_name
        else:
            return None
