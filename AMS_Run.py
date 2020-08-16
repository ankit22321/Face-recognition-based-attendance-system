import tkinter as tk
from tkinter import *
import cv2
import csv
import os
import numpy as np
from PIL import Image,ImageTk
import pandas as pd
import datetime
import time
from python_packages import admin_panel, manually_fill, choose_subject
import emoji 

#####Window is our Main frame of system
window = tk.Tk()
window.title("Real Time Facial Based Attendance System")

window.geometry('1280x720')
window.configure(background='white')


def clear():
    txt.delete(first=0, last=22)

def clear1():
    txt2.delete(first=0, last=22)



##Error screen2
def del_sc2():
    sc2.destroy()
def err_screen1():
    global sc2
    sc2 = tk.Tk()
    sc2.geometry('300x100')
    sc2.iconbitmap('UEMK.ico')
    sc2.title('Warning!!')
    sc2.configure(background='white')
    Label(sc2,text='Please enter your subject name!!!',fg='black',bg='white',font=('times', 16, ' bold ')).pack()
    Button(sc2,text='OK',command=del_sc2,fg="black"  ,bg="white"  ,width=9  ,height=1, activebackground = "lawn green" ,font=('times', 15, ' bold ')).place(x=90,y= 50)
	

	




def err_screen():
    global sc1
    sc1 = tk.Tk()
    sc1.geometry('300x100')
    sc1.iconbitmap('UEMK.ico')
    sc1.title('Warning!!')
    sc1.configure(background='white')
    Label(sc1,text='Enrollment & Name required!!!',fg='black',bg='white',font=('times', 16, ' bold ')).pack()
    Button(sc1,text='OK',command=del_sc1,fg="black"  ,bg="white"  ,width=9  ,height=1, activebackground = "lawn green" ,font=('times', 15, ' bold ')).place(x=90,y= 50)
	
	
def del_sc1():
    sc1.destroy()



###For take images for datasets
def take_img():
    l1 = txt.get()
    l2 = txt2.get()
    if l1 == '':
        err_screen()
    elif l2 == '':
        err_screen()
    else:
        try:
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            Enrollment = txt.get()
            Name = txt2.get()
            sampleNum = 0
            while (True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # incrementing sample number
                    sampleNum = sampleNum + 1
                    # saving the captured face in the dataset folder
                    cv2.imwrite("TrainingImage/ " + Name + "." + Enrollment + '.' + str(sampleNum) + ".jpg",
                                gray[y:y + h, x:x + w])
                    cv2.imshow('Frame', img)
                # wait for 100 miliseconds
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # break if the sample number is morethan 100
                elif sampleNum > 200:
                    break
            cam.release()
            cv2.destroyAllWindows()
            ts = time.time()
            Date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            Time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            row = [Enrollment, Name, Date, Time]
            with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile, delimiter=',')
                writer.writerow(row)
                csvFile.close()
            res = "Images Saved for Enrollment : " + Enrollment + " Name : " + Name
            Notification.configure(text=res, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
            Notification.place(x=300, y=400)
        except FileExistsError as F:
            f = 'Student Data already exists'
            Notification.configure(text=f, bg="lawn green", width=21)
            Notification.place(x=300, y=400)
			#clear()
			#clear1()
			
			

			
###For train the model
def trainimg():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    global detector
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    try:
        global faces,Id
        faces, Id = getImagesAndLabels("TrainingImage")
    except Exception as e:
        l='please make "TrainingImage" folder & put Images'
        Notification.configure(text=l, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
        Notification.place(x=300, y=400)

    recognizer.train(faces, np.array(Id))
    try:
        recognizer.save("TrainingImageLabel\Trainner.yml")
    except Exception as e:
        q='Please make "TrainingImageLabel" folder'
        Notification.configure(text=q, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
        Notification.place(x=300, y=400)

    res = "Model Trained"  # +",".join(str(f) for f in Id)
    Notification.configure(text=res, bg="SpringGreen3", width=50, font=('times', 18, 'bold'))
    Notification.place(x=300, y=400)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faceSamples = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image

        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)
        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    return faceSamples, Ids

	
def on_closing():
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()
#window.protocol("WM_DELETE_WINDOW", on_closing)

#white Should be replaced by cyan
message = tk.Label(window, text="Real Time Facial Based Attendance System", bg="white", fg="black", width=50,
                   height=3, font=('times', 30, 'bold '))

message.place(x=80, y=20)

Notification = tk.Label(window, text="All things good", bg="Green", fg="white", width=15,
                      height=3, font=('times', 17, 'bold'))

#lbl = tk.Label(window, text="Enter Enrollment", width=20, height=2, fg="black", bg="white", font=('times', 15, ' bold '))
#lbl.place(x=200, y=200)	
	
	
	
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
window.iconbitmap('UEMK.ico')



def testVal(inStr,acttyp):
    if acttyp == '1': #insert
        if not inStr.isdigit():
            return False
    return True

lbl = tk.Label(window, text="Enter Enrollment", width=20, height=2, fg="black", bg="white", font=('times', 15, ' bold '))
#lbl.place(x=200, y=200)	
lbl.place(x=500, y=240)

txt = tk.Entry(window, validate="key", width=20, bg="white", fg="black", font=('times', 25, ' bold '))
txt['validatecommand'] = (txt.register(testVal),'%P','%d')
txt.place(x=500, y=210)

lbl2 = tk.Label(window, text="Enter Name", width=20, fg="black", bg="white", height=2, font=('times', 15, ' bold '))
#lbl2.place(x=200, y=300)
lbl2.place(x=500, y=340)
txt2 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 25, ' bold '))
txt2.place(x=500, y=310)

#clearButton = tk.Button(window, text="Clear",command=clear,fg="black"  ,bg="white"  ,width=10  ,height=1 ,activebackground = "lawn green" ,font=('times', 15, ' bold '))
#clearButton.place(x=950, y=210)

#clearButton1 = tk.Button(window, text="Clear",command=clear1,fg="black"  ,bg="white"  ,width=10 ,height=1, activebackground = "lawn green" ,font=('times', 15, ' bold '))
#clearButton1.place(x=950, y=310)

#AP = tk.Button(window, text="Check Register Students",command=admin_panel.admin_panel,fg="black"  ,bg="white"  ,width=19 ,height=1, activebackground = "lawn green" ,font=('times', 15, ' bold '))
#AP.place(x=990, y=410)

takeImg = tk.Button(window, text="Take Images",command=take_img,fg="black"  ,bg="white"  ,width=18  ,height=3, activebackground = "lawn green" ,font=('times', 15, ' bold '))
#takeImg.place(x=90, y=500)
takeImg.place(x=900, y=200)

trainImg = tk.Button(window, text="Train Images",fg="black",command=trainimg ,bg="white"  ,width=18  ,height=3, activebackground = "lawn green" ,font=('times', 15, ' bold '))
#trainImg.place(x=390, y=500)
trainImg.place(x=900, y=300)

FA = tk.Button(window, text="Automatic Attendace",fg="black",command=choose_subject.subjectchoose  ,bg="white"  ,width=18  ,height=3, activebackground = "lawn green" ,font=('times', 15, ' bold '))
#FA.place(x=690, y=500)
FA.place(x=150, y=200)

quitWindow = tk.Button(window, text="Manually Fill Attendance", command=manually_fill.manually_fill  ,fg="black"  ,bg="white"  ,width=18  ,height=3, activebackground = "lawn green" ,font=('times', 15, ' bold '))
#quitWindow.place(x=990, y=500)
quitWindow.place(x=150, y=300)

lbl3 = tk.Label(window, text="Created with ❤️ by Ajay,Ankit,Hasnain,Prince & Rudra", width=50, fg="black", bg="white", height=2, font=('times', 15, ' bold '))
lbl3.place(x=400, y=500)


window.mainloop()