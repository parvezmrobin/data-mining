import os
from tkinter import *
from tkinter.filedialog import askdirectory, asksaveasfile, askopenfilename

import numpy as np
from matplotlib.pyplot import imread
from pandas import DataFrame, read_csv

true, false, null = True, False, None


class Similarity:

    def __init__(self):
        self.root = Tk()
        self.root.title("Data Mining")
        self.status_text = StringVar()
        self.canvas = null

        self.train_dir = null
        self.train_data = null
        self.file_names = null

        self.means = []
        self.medians = []
        self.modes = []
        self.mid_ranges = []
        self.ranges = []
        self.iqrs = []
        self.std_dev = []
        self.df = null

        self.query_image = null
        self.feature_vector = null
        self.nearest_images = null

        self.create_gui()

    def create_gui(self):
        config = {"padx": 10, "pady": 10, "text": "Output Area"}
        output_frame = LabelFrame(self.root, **config)

        self.canvas = Canvas(output_frame)
        self.canvas.pack()
        output_frame.pack()

        config["text"] = "Options"
        btn_frame = LabelFrame(self.root, **config)
        btn_frame.pack()

        config["text"] = "Load Train Data"
        btn_load_train = Button(btn_frame, **config, command=self.read_train_data)
        # btn_load_train.bind("<ButtonPress>", lambda event: print("asdfb"))
        btn_load_train.pack(side=LEFT)

        config["text"] = "Extract & Save Features"
        btn_extract_features = Button(btn_frame, **config, command=self.extract_features)
        btn_extract_features.pack(side=LEFT)

        config["text"] = "Load Features"
        btn_load_features = Button(btn_frame, **config, command=self.load_features)
        btn_load_features.pack(side=LEFT)

        config["text"] = "Select Query Image"
        btn_select_img = Button(btn_frame, **config, command=self.load_query_image)
        btn_select_img.pack(side=LEFT)

        config["text"] = "Show Similar Images"
        btn_show_img = Button(btn_frame, **config, command=self.show_similar_images)
        btn_show_img.pack(side=LEFT)

        status_label = Label(self.root, pady=5, textvariable=self.status_text)
        status_label.pack(side=BOTTOM)

    def start(self):
        self.root.mainloop()

    def read_train_data(self):
        self.train_dir = askdirectory()
        self.status_text.set('Reading from directory "' + self.train_dir + '"')
        self.root.update_idletasks()
        if self.train_dir == '':
            self.status_text.set("No directory selected")
            return

        file_names = os.listdir(self.train_dir)
        n = len(file_names)
        self.file_names = [self.train_dir + os.sep + file_name for file_name in file_names]
        images = [imread(file_name) for file_name in self.file_names]
        images = np.array(images)
        gray_images = np.mean(images, axis=3)
        self.train_data = np.reshape(gray_images, (n, -1))

        self.status_text.set('Successfully read from directory "' + self.train_dir + '"')
        # Reset df as data is changed
        self.df = null

    def extract_features(self):
        if self.train_data is null:
            self.status_text.set("No data is selected for extracting feature.")
            return

        m, n = self.train_data.shape
        self.means = np.mean(self.train_data, axis=1)
        self.medians = np.median(self.train_data, axis=1)
        self.modes = self._modes(self.train_data)
        maxes = np.max(self.train_data, axis=1)
        mins = np.min(self.train_data, axis=1)
        self.mid_ranges = (maxes + mins) * .5
        self.ranges = (maxes - mins)
        sorted_data = np.sort(self.train_data, axis=1)
        q_size = int(n / 4)
        q1 = sorted_data[:, q_size]
        q3 = sorted_data[:, 3 * q_size]
        self.iqrs = q3 - q1
        self.std_dev = np.std(self.train_data, axis=1)

        assert self.means.shape == self.medians.shape == self.modes.shape == self.mid_ranges.shape
        assert self.mid_ranges.shape == self.ranges.shape == self.iqrs.shape == self.std_dev.shape == (m,)

        data = np.array([self.means, self.medians, self.modes, self.mid_ranges,
                         self.ranges, self.iqrs, self.std_dev, self.file_names])
        columns = ["Mean", "Median", "Mode", "MidRange", "Range", "IQR", "StdDev", "FileName"]
        self.df = DataFrame(data.T, columns=columns)

        self.status_text.set("Features extracted")

        self.save_extracted_features()

    def save_extracted_features(self):
        if self.df is null:
            return
        filetypes = (("CSV File", "*.csv"), ("All Files", "*.*"))

        handle = null
        try:
            handle = asksaveasfile(defaultextension=".csv", filetypes=filetypes)
            if handle is null:
                self.status_text.set("No file is selected for writing output.")
                return
            self.df.to_csv(handle)
            self.status_text.set('Features are written to "' + handle.name + '"')
        except PermissionError as err:
            self.status_text.set('Could not write to file "' + err.filename + '"')
        finally:
            if handle is not null:
                handle.close()

    def load_features(self):
        filetypes = (("CSV File", "*.csv"), ("All Files", "*.*"))
        feature_file = askopenfilename(filetypes=filetypes)
        if feature_file == '':
            self.status_text.set("No file selected.")
            return
        self.df = read_csv(feature_file, index_col=0)
        self.status_text.set('Successfully read features from "{}".'.format(feature_file))

    def load_query_image(self):
        image_file = askopenfilename()
        if image_file == '':
            self.status_text.set("No image selected.")
            return
        self.query_image = imread(image_file)
        self.query_image = np.mean(self.query_image, axis=2).reshape((-1))
        self.status_text.set('Successfully read query image : "{}"'.format(image_file))

    def show_similar_images(self):
        if self.df is null:
            self.status_text.set("Load the features first.")
            return
        if self.query_image is null:
            self.status_text.set("Load the query image first.")
            return

        self.feature_vector = [
            np.mean(self.query_image),
            np.median(self.query_image),
            Similarity._mode1d(self.query_image)
        ]
        max_val = np.max(self.query_image)
        min_val = np.min(self.query_image)
        self.feature_vector += [(max_val + min_val) * .5, (max_val - min_val)]

        sorted_values = np.sort(self.query_image)
        q_size = int(sorted_values.shape[0] / 4)
        q1 = sorted_values[q_size]
        q3 = sorted_values[q_size * 3]
        self.feature_vector.append(q3 - q1)

        self.feature_vector.append(np.std(self.query_image))
        assert len(self.feature_vector) == len(self.df.columns) - 1

        self.find_nearest_images()

    def find_nearest_images(self, pick=3):
        def distance(df: DataFrame):
            cols = ['Mean', 'Median', 'Mode', 'MidRange', 'Range', 'IQR', 'StdDev']
            df = df[cols]
            for i, col in enumerate(cols):
                df[col] = (df[col] - self.feature_vector[i]) ** 2
            diff_col = df.sum(axis=1)
            return diff_col

        self.df: DataFrame
        temp_df = self.df.assign(Diff=distance).sort_values('Diff')
        self.nearest_images = temp_df.FileName[:pick]

        self.canvas: Canvas
        img = PhotoImage(file=self.nearest_images.iloc[1])
        self.canvas.create_image(50, 50, anchor=NE, image=img)

    @staticmethod
    def _mode1d(arr):
        u, c = np.unique(arr, return_counts=true)
        max_index = np.argmax(c)
        return u[max_index]

    @staticmethod
    def _modes(train_data):
        modes = np.apply_along_axis(Similarity._mode1d, 1, train_data)
        return modes


def main():
    Similarity().start()


main()
