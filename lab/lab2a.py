import os
from tkinter import *
from tkinter.filedialog import askdirectory, asksaveasfile

import numpy as np
from matplotlib.pyplot import imread

true, false, null = True, False, None


class Similarity:

    def __init__(self):
        self.root = Tk(className="Similarity")
        self.root.title("Data Mining")
        self.status_text = StringVar()

        self.train_dir = null
        self.train_data = []

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
        btn_load_features = Button(btn_frame, **config)
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

    def extract_features(self):
        handle = asksaveasfile(defaultextension=".csv", filetypes=(("CSV File", "*.csv"), ("All Files", "*.*")))
        handle.write("a,b,c\n1,2,4")
        handle.close()


def main():
    Similarity().start()


main()
