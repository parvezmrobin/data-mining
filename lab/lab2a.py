from tkinter import *

root = Tk(className="Similarity")

btn_frame = Frame(root)
btn_frame.pack()

config = {
    "text": "Load Train Data",
    "padx": 10,
    "pady": 10,
}
btn_load_train = Button(btn_frame, **config)
btn_load_train.bind("<ButtonPress>", lambda event: print("asdfb"))
btn_load_train.pack(side=LEFT)

config["text"] = "Extract & Save Features"
btn_extract_features = Button(btn_frame, **config)
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

root.mainloop()
