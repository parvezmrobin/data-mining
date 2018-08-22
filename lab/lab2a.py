import os
from tkinter import *
from tkinter.filedialog import askdirectory, asksaveasfile, askopenfile

import numpy as np
from matplotlib.pyplot import imread
from pandas import DataFrame, read_csv

true, false, null = True, False, None


class Similarity:

    def __init__(self):
        self.root = Tk(className="Similarity")
        self.root.title("Data Mining")
        self.status_text = StringVar()

        self.train_dir = null
        self.train_data = null

        self.means = []
        self.medians = []
        self.modes = []
        self.mid_ranges = []
        self.ranges = []
        self.iqrs = []
        self.std_dev = []
        self.df = null

        self.create_gui()

    def create_gui(self):
        config = {"padx": 10, "pady": 10, "text": "Output Area"}
        output_frame = LabelFrame(self.root, **config)

        canvas = Canvas(output_frame)
        canvas.pack()
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
        btn_select_img = Button(btn_frame, **config)
        btn_select_img.pack(side=LEFT)

        config["text"] = "Show Similar Images"
        btn_show_img = Button(btn_frame, **config)
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
        file_names = [self.train_dir + os.sep + file_name for file_name in file_names]
        images = [imread(file_name) for file_name in file_names]
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

        data = np.array([self.means, self.medians, self.modes,
                         self.mid_ranges, self.ranges, self.iqrs, self.std_dev])
        columns = ["Mean", "Median", "Mode", "MidRange", "Range", "IQR", "StdDev"]
        self.df = DataFrame(data.T, columns=columns)

        self.status_text.set("Features extracted")

        self.save_extracted_features()

    def load_features(self):
        filetypes = (("CSV File", "*.csv"), ("All Files", "*.*"))
        feature_file = askopenfile(filetypes=filetypes)
        if feature_file == '':
            self.status_text.set("No file selected.")
            return
        self.df = read_csv(feature_file)
        self.status_text.set('Successfully read features from "{}".'.format(feature_file.name))
        feature_file.close()

    def save_extracted_features(self):
        self.df: DataFrame
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
            self.status_text.set('Could not open file "' + err.filename + '"')
        finally:
            if handle is not null:
                handle.close()

    @staticmethod
    def _modes(train_data):
        def mode1d(arr):
            u, c = np.unique(arr, return_counts=true)
            max_index = np.argmax(c)
            return u[max_index]

        modes = np.apply_along_axis(mode1d, 1, train_data)

        return modes


def main():
    Similarity().start()


main()
