#!/usr/bin/env python3

import cv2
import tkinter as tk
import tkinter.font as tkFont
import tkinter.filedialog as filedialog
from PIL import Image, ImageTk

from predict_emotionandface_image import *
from predict_emotionandface_video import *
from train_face_regonise import *

obj = tk.Tk(className='A09')
obj.geometry('800x480')
tkfont = tkFont.Font(family='Font Awesome 5 Free',size=18, weight=tkFont.NORMAL)
# create the main frame
main_frame = tk.Frame(obj, background='black')
left_frame = tk.Frame(main_frame, background='yellow')
right_frame = tk.Frame(main_frame, background='white')
bottom_frame = tk.Frame(main_frame, background='blue')

main_frame.place(x=0, y=0, anchor='nw', width=800, height=480)
left_frame.place(x=0, y=0, anchor='nw', width=400, height=460)
right_frame.place(x=400, y=0, anchor='nw', width=400, height=460)
bottom_frame.place(x=0, y=460, anchor='nw', width=800, height=20)

canvas = tk.Canvas(left_frame)
canvas.place(x=0, y=0, anchor='nw', width=400, height=460)
# canvas.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)

# lbutton = tk.Button(left_frame, text='left frame')
# lbutton.pack()

# rbutton = tk.Button(right_frame, text='right frame')
# rbutton.grid()
# label_name = tk.Label(right_frame, text='姓名')
rtext = tk.Text(right_frame, cursor=None, font=tkfont)
rtext.pack()

# callback function
def upload_photo():
    rtext.delete('1.0', tk.END)
    if os.name =='posix':
        File = filedialog.askopenfilename(parent=obj, initialdir='~/Pictures', title='Choose an image')
    if os.name == 'nt':
        File = filedialog.askopenfilename(parent=obj, initialdir='~/Pictures', title='Choose an image')
    # 调用函数传入图片
    img, emotion_result, result = photoface_gg(File)
    (angry, disgust, fear, happy, sad, surprise, neutral) = emotion_result
    rtext.insert(tk.INSERT, 'The Probability of each motion:\n')
    rtext.insert(tk.INSERT, 'Angry: ' + str(angry) + '\n')
    rtext.insert(tk.INSERT, 'Disgust: ' + str(disgust) + '\n')
    rtext.insert(tk.INSERT, 'Fear: ' + str(fear) + '\n')
    rtext.insert(tk.INSERT, 'Happy: ' + str(happy) + '\n')
    rtext.insert(tk.INSERT, 'Sad: ' + str(sad) + '\n')
    rtext.insert(tk.INSERT, 'Surprise: ' + str(surprise) + '\n')
    rtext.insert(tk.INSERT, 'Neutral: ' + str(neutral) + '\n')
    # img = img.reshape((400, 460))
    img = Image.fromarray(img)
    img = img.resize((400, 460), Image.ANTIALIAS)
    filename = ImageTk.PhotoImage(image=img)
    canvas.image = filename
    canvas.create_image(0, 0, anchor='nw', image=filename)

    # rtext.insert(tk.END, 'Name: ' + name)

# camera
def open_camera():
    creat_video()

def train(win, entry):
    name = entry.get()
    win.destroy()
    train_model(name)

# Add new facce
def addnewface():
    win = tk.Tk(className='Add new face')
    win.geometry('400x240')
    label = tk.Label(win, text='Name')
    label.grid()
    entry = tk.Entry(win)
    entry.grid()
    # name = entry.get()
    button = tk.Button(win, text='Ok', command=lambda: train(win, entry))
    button.grid()
    # train_model(name)

# bottom_frame's buttons
bbutton = tk.Button(bottom_frame, text='upload photo', command=upload_photo)
bbutton.place(x=0, y=0, anchor='nw', width=200, height=20)
bbutton_1 = tk.Button(bottom_frame, text='open camera', command=open_camera)
bbutton_1.place(x=200, y=0, anchor='nw', width=200, height=20)
bbuton_2 = tk.Button(bottom_frame, text='Add New Face', command=addnewface)
bbuton_2.place(x=400, y=0, anchor='nw', width=200, height=20)

obj.mainloop()
