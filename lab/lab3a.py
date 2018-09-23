import os
from tkinter import Tk, StringVar, LabelFrame, Button, LEFT, Label, BOTTOM, HORIZONTAL, BOTH, Frame
from tkinter.filedialog import askdirectory, asksaveasfile, askopenfilename
from tkinter.ttk import Progressbar

import numpy as np
from matplotlib.pyplot import imread
from skimage.feature import local_binary_pattern
from sklearn.tree import DecisionTreeClassifier

true, false, null = True, False, None
eps = np.finfo(np.float32).eps


class DecisionTree:
    classes = ["apple", "car", "cow", "cup", "dog", "horse", "pear", "tomato", ]

    progress_bar: Progressbar
    status_text: StringVar

    train_data: np.ndarray
    train_file_names: list
    train_features: list
    train_classes: list

    test_data: np.ndarray
    test_file_names: list
    test_features: list
    test_classes: list

    def __init__(self):
        self.root = Tk()
        self.root.title("Data Mining")

        self.create_gui()

    def create_gui(self):
        config = {"padx": 10, "pady": 10, "text": "Options"}

        btn_frame = LabelFrame(self.root, **config)
        btn_frame.pack()

        config["text"] = "Load Train Data"
        btn_load_train = Button(btn_frame, **config, command=self.read_train_data)
        btn_load_train.pack(side=LEFT)

        config["text"] = "Extract & Save Features"
        btn_extract_features = Button(btn_frame, **config, command=self.extract_features)
        btn_extract_features.pack(side=LEFT)

        config["text"] = "Load Features"
        btn_load_features = Button(btn_frame, **config, command=self.load_features)
        btn_load_features.pack(side=LEFT)

        config["text"] = "Load Test Data"
        btn_select_img = Button(btn_frame, **config, command=self.load_test_folder)
        btn_select_img.pack(side=LEFT)

        config["text"] = "Classify"
        btn_show_img = Button(btn_frame, **config, command=self.classify)
        btn_show_img.pack(side=LEFT)

        bottom_frame = Frame(self.root, padx=10, pady=10)
        bottom_frame.pack(side=BOTTOM, fill=BOTH)

        self.status_text = StringVar()
        status_label = Label(bottom_frame, pady=5, textvariable=self.status_text)
        status_label.pack()

        self.progress_bar = Progressbar(bottom_frame, orient=HORIZONTAL, mode='determinate', )
        self.progress_bar.pack(fill=BOTH)

        self.root.mainloop()

    def read_train_data(self):
        train_dir = askdirectory()
        if train_dir == '':
            self.status_text.set("No directory selected")
            return

        self.status_text.set('Reading from directory "' + train_dir + '"')
        self.root.update_idletasks()

        self.train_file_names = os.listdir(train_dir)
        n = len(self.train_file_names)

        images = []
        self.progress_bar['value'] = 0
        self.root.update_idletasks()
        for file_name in self.train_file_names:
            images.append(imread(train_dir + os.sep + file_name))
            self.progress_bar.step(100 / n)
            self.root.update_idletasks()

        self.progress_bar['value'] = 100

        images = np.array(images)
        self.train_data = np.mean(images, axis=3)

        self.status_text.set('Successfully read from directory "{}"'.format(train_dir))
        # Reset features as data is changed
        self.train_features = null

    def extract_features(self):
        if self.train_data is null:
            self.status_text.set("No data is selected for extracting feature.")
            return

        self.status_text.set('Extracting features from training data.')
        self.progress_bar['value'] = 0

        self.progress_bar['value'] = 0
        self.root.update_idletasks()

        n, w, h = self.train_data.shape
        self.train_features = []
        for i in range(n):
            feature = DecisionTree._get_feature_and_class(
                self.train_data[i], self.train_file_names[i]
            )
            self.train_features.append(feature)
            self.progress_bar.step(100 / n)
            self.root.update_idletasks()

        self.progress_bar['value'] = 100

        self.status_text.set("Features extracted from training data.")
        self.root.update_idletasks()

        self.save_extracted_features()

    def save_extracted_features(self):
        if self.train_features is null:
            return
        file_types = (("CSV File", "*.csv"), ("All Files", "*.*"))

        handle = null
        try:
            handle = asksaveasfile(defaultextension=".csv", filetypes=file_types)
            if handle is null:
                self.status_text.set("No file is selected for writing output.")
                return
            np.savetxt(handle.name, self.train_features, fmt=['%.10e']*10 + ['%i'], delimiter=',')
            self.status_text.set('Features are written to "' + handle.name + '"')
        except PermissionError as err:
            self.status_text.set('Could not write to file "' + err.filename + '"')
        finally:
            if handle is not null:
                handle.close()

    def load_features(self):
        file_types = (("CSV File", "*.csv"), ("All Files", "*.*"))
        feature_file = askopenfilename(filetypes=file_types)
        if feature_file == '':
            self.status_text.set("No file selected.")
            return
        self.train_features = np.genfromtxt(feature_file, delimiter=',')
        self.status_text.set('Successfully read features from "{}".'.format(feature_file))

    def load_test_folder(self):
        test_dir = askdirectory()
        if test_dir == '':
            self.status_text.set("No directory selected")
            return

        self.status_text.set('Reading from directory "' + test_dir + '"')
        self.root.update_idletasks()

        file_names = os.listdir(test_dir)
        n = len(file_names)
        self.test_file_names = [test_dir + os.sep + file_name for file_name in file_names]

        images = []
        self.progress_bar['value'] = 0
        self.root.update_idletasks()
        for file_name in self.test_file_names:
            images.append(imread(file_name))
            self.progress_bar.step(100 / n)
            self.root.update_idletasks()

        self.progress_bar['value'] = 100

        images = np.array(images)
        self.test_data = np.mean(images, axis=3)

        self.status_text.set('Successfully read from directory "{}"'.format(test_dir))
        # Reset features as data is changed
        self.test_features = null

        self.extract_test_features()

    def extract_test_features(self):
        if self.test_data is null:
            self.status_text.set("No data is selected for extracting feature.")
            return

        self.status_text.set('Extracting features from testing data.')
        self.progress_bar['value'] = 0
        self.root.update_idletasks()

        n, w, h = self.test_data.shape
        self.test_features = []
        for i in range(n):
            feature = DecisionTree._get_feature(self.test_data[i])
            self.test_features.append(feature)
            self.progress_bar.step(100 / n)
            self.root.update_idletasks()

        self.progress_bar['value'] = 100

        self.status_text.set("Features extracted from testing data.")
        self.root.update_idletasks()

    def classify(self):
        def transform(row):
            i, cls = row
            return [i + 1, self.test_file_names[i], DecisionTree.classes[cls]]
        dt = DecisionTreeClassifier()
        features = self.train_features[:, :-1]
        classes = self.train_features[:, -1]
        classes = classes.astype('int')
        dt.fit(features, classes)
        test_classes = dt.predict(self.test_features)
        output = list(map(transform, enumerate(test_classes)))
        file = asksaveasfile()
        np.savetxt(file.name, output, fmt=['%s', '%s', '%s'], delimiter=',')
        self.status_text.set("Test classes are written to '{}'".format(file.name))

    @staticmethod
    def _get_class(file_name):
        for i, cls in enumerate(DecisionTree.classes):
            if file_name.startswith(cls):
                return i
        raise RuntimeError("No class found")

    @staticmethod
    def _get_feature_and_class(image, file_name, num_points=8, radius=1, combine=true):
        hist = DecisionTree._get_feature(image, num_points, radius)
        cls = DecisionTree._get_class(file_name)
        if combine:
            return np.append(hist, [cls])
        return hist, cls

    @staticmethod
    def _get_feature(image, num_points=8, radius=1):
        lbp = local_binary_pattern(image, num_points, radius, 'uniform')
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, num_points + 3),
            range=(0, num_points + 2)
        )
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist


def main():
    DecisionTree()


main()
