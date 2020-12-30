import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilename, asksaveasfile, asksaveasfilename
from PIL import ImageTk, Image, ImageDraw
import numpy as np
import cv2
import pytesseract
import csv
from corner import white_box

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'F:/c++opencv/finalProject/tesseract/tesseract.exe'
window = tk.Tk()
window.geometry('1080x600')
window.resizable(width=False, height=False)
window.columnconfigure([0, 1], weight=1)
window.rowconfigure([0, 1], weight=1)
# global path
path = None
save_face = None
save_text = None
save_img = None


def save_image():
    filename = asksaveasfile(mode='w',
                             defaultextension='.jpg',
                             filetypes=[('image files', ('.jpg'))])
    # filename = asksaveasfile(mode='w',
    #                          defaultextension='.csv',
    #                          filetypes=[('image files', ('.csv'))])
    if not filename:
        return
    global save_face
    save_face.save(filename)


def file_save():
    # f = asksaveasfile(mode='wb', defaultextension=".txt")
    # if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
    #     return
    # global save_text
    # a = save_text.encode('utf-8')
    # f.write(a)
    # f.close()  # `()` was missing.
    a = white_box().black_box(cv2.imread(path))
    # university name
    uName = white_box().toText(a.copy()[27:27 + 58 - 27,
                                        103:103 + 570 - 103])[:-2]

    # name
    pName = white_box().toText(a.copy()[127:127 + 173 - 127,
                                        273:273 + 472 - 273])[:-2]

    # birthday
    pBirthDay = white_box().toText(a.copy()[165:209, 305:501])[:-2]

    # major
    pMajor = white_box().toText(a.copy()[205:244, 301:550])[:-2]

    # year of admission
    pYear = white_box().toText(a.copy()[261:290, 332:410])[:-2]

    # ID
    pId = white_box().toText(a.copy()[374:433, 36:200])[:-2]

    with open('cardInfo.csv', mode='w', encoding='utf-8-sig') as csv_file:
        fieldnames = [
            'university', 'name', 'birthday', 'major', 'year of admission',
            'ID'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow({
            'university': uName,
            'name': pName,
            'birthday': pBirthDay,
            'major': pMajor,
            'year of admission': pYear,
            'ID': pId
        })


def text_extraction(path):
    # large = cv2.imread(path)
    large = white_box().black_box(cv2.imread(path))
    text = pytesseract.image_to_string(large,
                                       lang='vie',
                                       config='--psm 1 --oem 3')
    return text


def face(image):
    face_classifier = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')

    # Load our image then convert it to grayscale
    # image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Our classifier returns the ROI of the detected face as a tuple
    # It stores the top left coordinate and the bottom right coordiantes
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # When no faces detected, face_classifier returns and empty tuple
    if faces is ():
        print("No faces found")
        return (False, False)

    # We iterate through our faces array and draw a rectangle
    # over each face in faces
    for (x, y, w, h) in faces:
        x = x - 25  # Padding trick to take the whole face not just Haarcascades points
        y = y - 40  # Same here...
        c = image[y:(y + h + 70), x:(x + w + 50)]
        cv2.rectangle(image, (x, y), (x + w + 50, y + h + 70), (27, 200, 10),
                      2)
        return (c, True)


# open image folder
def Open_image_folder():
    filename = askopenfilename(title='open')
    return filename


# function open image from file
def Img_from_file(path):
    load = Image.open(path)
    # frm1.update()
    w = frm1.winfo_width()
    h = frm1.winfo_height()
    load = load.resize((w - 3, h - 3))
    render = ImageTk.PhotoImage(load)
    print((w, h))
    return render


# load image frame
frm1 = tk.Label(master=window,
                background='#242424',
                relief=tk.RAISED,
                foreground='#f5f5f5',
                font=("Courier", 16),
                text='load image from file',
                borderwidth=1)

# show text detection
frm2 = tk.Label(master=window,
                background='#242424',
                text='Perspective transform',
                foreground='#f5f5f5',
                font=("Courier", 16),
                relief=tk.RAISED,
                borderwidth=1)

# show extract face image
frm3 = tk.Label(master=window,
                background='#242424',
                foreground='#f5f5f5',
                font=("Courier", 16),
                text='extract face',
                relief=tk.RAISED,
                borderwidth=1)

# show extracted text
frm4 = tk.Text(master=window,
               font=("Courier", 16),
               background='#242424',
               foreground='#f5f5f5',
               width=1,
               height=1,
               relief=tk.RAISED,
               borderwidth=1)
frm4.insert('1.0', 'extracted text')
# set content of each frame

# frame 1: load image
frm1.grid(row=0, column=0, sticky='nswe', padx=3, pady=3)

# frame 2: load text detection
frm2.grid(row=0, column=1, sticky='nswe', padx=3, pady=3)

# frame 3: load face image
frm3.grid(row=1, column=0, sticky='nswe', padx=3, pady=3)

# frame 4: load extracted text
frm4.grid(row=1, column=1, sticky='nswe', padx=3, pady=3)


# show image in frame 1
def frm1_image():
    global path
    path = Open_image_folder()
    img = Img_from_file(path)
    img_lbl = tk.Label(window, image=img)
    img_lbl.img = img
    img_lbl.place(x=0, y=0)
    global save_img
    save_img = img
    #face

    # img=cv2.imread(path)
    # img3=face(img)
    # img3=cv2.resize(img3,(0,0),fx=0.5,fy=0.5)
    # b,g,r = cv2.split(img3)
    # img3 = cv2.merge((r,g,b))
    # im = Image.fromarray(img3)
    # imgtk = ImageTk.PhotoImage(image=im)
    # img_lbl3=tk.Label(window,image=imgtk)
    # img_lbl3.img=imgtk
    # img_lbl3.place(x=0+int((540-imgtk.width())/2),y=300+int((300-imgtk.height())/2))


# extract face
def frm3_image():
    # read image
    # img = cv2.imread(path)
    im = white_box().black_box(cv2.imread(path))

    # get face
    img3, flag = face(im)
    if flag == False:
        frm3.configure(text='No face found       ')
        frm3.update()
    else:
        img3 = cv2.resize(img3, (0, 0), fx=1, fy=1)
        b, g, r = cv2.split(img3)
        img3 = cv2.merge((r, g, b))
        im = Image.fromarray(img3)
        imgtk = ImageTk.PhotoImage(image=im)
        img_lbl3 = tk.Label(window, image=imgtk)
        img_lbl3.img = imgtk
        img_lbl3.place(x=0 + int((540 - imgtk.width()) / 2),
                       y=300 + int((300 - imgtk.height()) / 2))
        global save_face
        save_face = im


def frm2_image():
    large = cv2.imread(path)

    # rgb = cv2.pyrDown(large)
    # small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    # _, bw = cv2.threshold(grad, 0.0, 255.0,
    #                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    # connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # # using RETR_EXTERNAL instead of RETR_CCOMP
    # contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL,
    #                                        cv2.CHAIN_APPROX_NONE)
    # #For opencv 3+ comment the previous line and uncomment the following line
    # #_, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # mask = np.zeros(bw.shape, dtype=np.uint8)

    # for idx in range(len(contours)):
    #     x, y, w, h = cv2.boundingRect(contours[idx])
    #     mask[y:y + h, x:x + w] = 0
    #     cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    #     r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)

    #     if r > 0.45 and w > 8 and h > 8:
    #         cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
    # h, w, c = rgb.shape
    # # width of current frame
    # w_cur = frm1.winfo_width() - 3
    # # height of current frame
    # h_cur = frm1.winfo_height() - 3

    # temp = cv2.resize(rgb, (540, 294), interpolation=cv2.INTER_AREA)
    temp = white_box().black_box(large)
    temp = cv2.resize(temp, (540, 294), interpolation=cv2.INTER_AREA)

    # reconstruct image
    b, g, r = cv2.split(temp)
    temp = cv2.merge((r, g, b))
    im = Image.fromarray(temp)
    imgtk = ImageTk.PhotoImage(image=im)
    img_lbl2 = tk.Label(window, image=imgtk)
    img_lbl2.img = imgtk
    img_lbl2.place(x=540, y=0)
    # return rgb


def frm4_text():
    frm4.delete(1.0, tk.END)
    frm4.insert(1.0, 'Please wait...')
    frm4.update()
    text = text_extraction(path)
    frm4.delete(1.0, tk.END)
    frm4.insert(1.0, text)
    global save_text
    save_text = text


# config
frm1.rowconfigure(0, weight=1)
frm1.columnconfigure(0, weight=1)
frm2.rowconfigure(1, weight=1)
frm2.columnconfigure(0, weight=1)
frm3.rowconfigure(1, weight=1)
frm3.columnconfigure(0, weight=1)
frm4.rowconfigure(1, weight=1)
frm4.columnconfigure(0, weight=1)

#set menu bar
menubar = tk.Menu(window)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu1 = tk.Menu(menubar, tearoff=0)
filemenu3 = tk.Menu(menubar, tearoff=0)
filemenu3.add_command(label='Save text', command=file_save)
filemenu3.add_command(label='Save face image', command=save_image)

filemenu.add_command(label='Open image...', command=frm1_image)
filemenu1.add_command(label='Extract face from image', command=frm3_image)
filemenu1.add_command(label='Perspective transform', command=frm2_image)
filemenu1.add_command(label='Show extracted text', command=frm4_text)
menubar.add_cascade(label='file', menu=filemenu)
menubar.add_cascade(label='function', menu=filemenu1)
menubar.add_cascade(label='save', menu=filemenu3)
window.config(menu=menubar)

window.mainloop()