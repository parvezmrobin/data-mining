from tkinter import Tk, Canvas

from PIL import Image
from PIL.ImageTk import PhotoImage

root = Tk()
root.title("dfslkj")
canvas = Canvas(root, width=500, height=500, bg='blue')
canvas.pack()
img = PhotoImage(Image.open("C:/Users/Parvez/Desktop/38917948_498869790552629_4732584812324323328_n.jpg"))
canvas.create_image(10, 10, image=img)
root.mainloop()
