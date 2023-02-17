from binascii import a2b_hqx
import curses
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import os
from numpy import int16

menuList = ['Menu:', 'Image Processing', 'Corner Detection', 'Face Detection','Edge Detection', 'Shape Detection','Exit']
imageCornerDetectList = ['Bear', 'Deer', 'Dog']
imageProcessingList = ['Grayscale', 'Inverse Grayscale', 'Grayscale High Contrast', 'Blur', 'Increase Contrast', 'Mean Filter', 'Median Filter']
faceRecList = ['Check First Image', 'Check Second Image']
        
def cornerDetectOpt(screen, cdSelect):
    screen.clear()
    h,w =screen.getmaxyx()
    for idx, row in enumerate(imageCornerDetectList):
        x = int16(w/2 - len(row)/2)
        y = int16(h/2 - len(imageCornerDetectList)/2 + idx-2) 
        if idx == cdSelect:
            screen.attron(curses.color_pair(1))
            screen.addstr(y,x,row)
            screen.attroff(curses.color_pair(1))
        else:
            screen.addstr(y,x,row)
    while(True):
        key = screen.getch()
        screen.clear()
        if key == curses.KEY_UP and cdSelect>0:
            cdSelect-=1
        elif key == curses.KEY_DOWN and cdSelect<len(imageCornerDetectList)-1:
            cdSelect+=1
        elif (key == curses.KEY_ENTER or key in [10,13]):
            screen.clear()
            # screen.addstr(0,0,"you pressed {}".format(menuList[selected]))
            if cdSelect==0:
                cornerDetect('bear.jpg')
            elif cdSelect==1:
                cornerDetect('deer.jpg')
            elif cdSelect==2:
                cornerDetect('dog.jpg')
            printMenu(screen, menuList, 0)
            

        for idx, row in enumerate(imageCornerDetectList):
            x = int16(w/2 - len(row)/2)
            y = int16(h/2 - len(imageCornerDetectList)/2 + idx-2) 
            if idx == cdSelect:
                screen.attron(curses.color_pair(1))
                screen.addstr(y,x,row)
                screen.attroff(curses.color_pair(1))
            else:
                screen.addstr(y,x,row)
            
        screen.refresh()

def imageProcOpt(screen, ipSelect):
    screen.clear()
    h,w =screen.getmaxyx()
    for idx, row in enumerate(imageProcessingList):
        x = int16(w/2 - len(row)/2)
        y = int16(h/2 - len(imageProcessingList)/2 + idx-2) 
        if idx == ipSelect:
            screen.attron(curses.color_pair(1))
            screen.addstr(y,x,row)
            screen.attroff(curses.color_pair(1))
        else:
            screen.addstr(y,x,row)
    while(True):
        key = screen.getch()
        screen.clear()
        if key == curses.KEY_UP and ipSelect>0:
            ipSelect-=1
        elif key == curses.KEY_DOWN and ipSelect<len(imageProcessingList)-1:
            ipSelect+=1
        elif (key == curses.KEY_ENTER or key in [10,13]):
            screen.clear()
            b,g,r = cv.split(image1)
            ksize=10
            if ipSelect==0:
                showResult('Grayscale',imagecv,'gray')
            elif ipSelect==1:
                _, inv_bin_thresh=cv.threshold(imagecv, 100, 255, cv.THRESH_BINARY_INV)
                showResult('Grayscale Inverse', inv_bin_thresh,'gray')
            elif ipSelect==2:
                nequ_gray=cv.equalizeHist(imagecv)
                showResult('Grayscale High Contrast',nequ_gray,'gray')
            elif ipSelect==3:
                blur = cv.blur(image,(10,10))
                showResult('Blur', blur)
            elif ipSelect==4:
                showResult('increased contrast', imageyuv)
            elif ipSelect==5:
                b_mean = mean_filter(b, ksize)
                g_mean = mean_filter(g, ksize)
                r_mean = mean_filter(r, ksize)
                res_mean_filter = cv.merge((r_mean, b_mean, g_mean))
                showResult('Mean Filter',res_mean_filter)
            elif ipSelect==6:
                b_median = median_filter(b, ksize)
                g_median = median_filter(g, ksize)
                r_median = median_filter(r, ksize)
                res_median_filter = cv.merge((r_median, b_median, g_median))
                showResult('Median Filter',res_median_filter)
            printMenu(screen, menuList, 0)
            

        for idx, row in enumerate(imageProcessingList):
            x = int16(w/2 - len(row)/2)
            y = int16(h/2 - len(imageProcessingList)/2 + idx-2) 
            if idx == ipSelect:
                screen.attron(curses.color_pair(1))
                screen.addstr(y,x,row)
                screen.attroff(curses.color_pair(1))
            else:
                screen.addstr(y,x,row)
            
        screen.refresh()

def faceRecOpt(screen, frSelect):
    screen.clear()
    h,w =screen.getmaxyx()
    for idx, row in enumerate(faceRecList):
        x = int16(w/2 - len(row)/2)
        y = int16(h/2 - len(faceRecList)/2 + idx-2) 
        if idx == frSelect:
            screen.attron(curses.color_pair(1))
            screen.addstr(y,x,row)
            screen.attroff(curses.color_pair(1))
        else:
            screen.addstr(y,x,row)
    while(True):
        key = screen.getch()
        screen.clear()
        if key == curses.KEY_UP and frSelect>0:
            frSelect-=1
        elif key == curses.KEY_DOWN and frSelect<len(imageCornerDetectList)-1:
            frSelect+=1
        elif (key == curses.KEY_ENTER or key in [10,13]):
            screen.clear()
            # screen.addstr(0,0,"you pressed {}".format(menuList[selected]))
            if frSelect==0:
                detecting_face(0)
            elif frSelect==1:
                detecting_face(1)
            printMenu(screen, menuList, 0)
            

        for idx, row in enumerate(faceRecList):
            x = int16(w/2 - len(row)/2)
            y = int16(h/2 - len(faceRecList)/2 + idx-2) 
            if idx == frSelect:
                screen.attron(curses.color_pair(1))
                screen.addstr(y,x,row)
                screen.attroff(curses.color_pair(1))
            else:
                screen.addstr(y,x,row)
            
        screen.refresh()

def printMenu(screen, text, selected):
    curses.curs_set(0)
    curses. init_pair(1,curses.COLOR_BLACK, curses.COLOR_WHITE)
    screen.clear()
    h,w =screen.getmaxyx()
    for idx, row in enumerate(text):
        x = int16(w/2 - len(row)/2)
        y = int16(h/2 - len(text)/2 + idx-2) 
        if idx == selected:
            screen.attron(curses.color_pair(1))
            screen.addstr(y,x,row)
            screen.attroff(curses.color_pair(1))
        else:
            screen.addstr(y,x,row)
    while(True):
        key = screen.getch()
        screen.clear()
        if key == curses.KEY_UP and selected>0:
            selected-=1
        elif key == curses.KEY_DOWN and selected<len(menuList)-1:
            selected+=1
        elif (key == curses.KEY_ENTER or key in [10,13]) and selected!=0:
            screen.clear()
            # screen.addstr(0,0,"you pressed {}".format(menuList[selected]))
            if selected ==1:
                imageProcOpt(screen, 0)
            elif selected==2:
                cornerDetectOpt(screen, 0)
            elif selected==3:
                faceRecOpt(screen,0)
            elif selected==4:
                edgeDetect()
            elif selected==5:
                shapeDetect()
            elif selected==6:
                exit()
            screen.refresh()
            screen.getch()
            screen.clear()

        for idx, row in enumerate(text):
            x = int16(w/2 - len(row)/2)
            y = int16(h/2 - len(text)/2 + idx-2) 
            if idx == selected:
                screen.attron(curses.color_pair(1))
                screen.addstr(y,x,row)
                screen.attroff(curses.color_pair(1))
            else:
                screen.addstr(y,x,row)
            
        screen.refresh()
    
def showRes(source, cmap=None):
    plt.imshow(source, cmap=cmap)
    plt.show()
def edgeDetect():
    image = cv.imread('./pororo.jpg')
    igray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    igray = cv.Canny(igray, 75, 150,0,3,50)
    showRes(igray)
    curses.wrapper(printMenu, menuList,0)
def shapeDetect():
    image = cv.imread('shapes.png')
    igray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, threshold = cv.threshold(igray, 151, 255, cv.THRESH_BINARY)
 
    _, contours, _ = cv.findContours(
        threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    i = 0
    
   
    for contour in contours:
        
        if i == 0:
            i = 1
            continue
    
        approx = cv.approxPolyDP(
            contour, 0.01 * cv.arcLength(contour, True), True)
        

        cv.drawContours(image, [contour], 0, (0, 0, 255), 5)
    
        M = cv.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
    
        
        if len(approx) == 3:
            cv.putText(image, 'Triangle', (x, y),
                        cv.QT_FONT_NORMAL, 0.6, (135, 25, 100), 2)
    
        elif len(approx) == 4:
            cv.putText(image, 'Quadrilateral', (x, y),
                        cv.QT_FONT_NORMAL, 0.6, (135, 25, 100), 2)
    
        elif len(approx) == 5:
            cv.putText(image, 'Pentagon', (x, y),
                        cv.QT_FONT_NORMAL, 0.6, (135, 25, 100), 2)
    
        elif len(approx) == 6:
            cv.putText(image, 'Hexagon', (x, y),
                        cv.QT_FONT_NORMAL, 0.6, (135, 25, 100), 2)
    
        else:
            cv.putText(image, 'circle', (x, y),
                        cv.QT_FONT_NORMAL, 0.6, (135, 25, 100), 2)

    cv.imshow('shapes', image)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
def cornerDetect(image):
    image = cv.imread(image)
    igray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    igray = np.float32(igray)

    haris_corner = cv.cornerHarris(igray, 2, 5, 0.04)
    without_subpix = image.copy()

    without_subpix[haris_corner > 0.01 * haris_corner.max()] =[0,0,255]

    # showRes(without_subpix, 'gray')

    _,thresh = cv.threshold(haris_corner, 0.001* haris_corner.max(), 255, 0)
    thresh = np.uint8(thresh)

    _,_,_, centroid =cv.connectedComponentsWithStats(thresh)

    centroid = np.float32(centroid)

    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 0.0001)

    enhanced_corner = cv.cornerSubPix(igray, centroid, (2,2), (-1,-1), criteria)

    with_subpix = image.copy()
    enhanced_corner = np.uint16(enhanced_corner)

    for corner in enhanced_corner:
        x,y = corner[:2]
        with_subpix[y, x]= [255,240, 31]
    showRes(with_subpix)

image = mpimg.imread('./ipSample/puppy.jpg')
image1 = cv.imread('./ipSample/puppy.jpg')
imageyuv = cv.cvtColor(image1, cv.COLOR_BGR2YUV)
imageyuv[:,:,0] = cv.equalizeHist(imageyuv[:,:,0])

# convert the YUV image back to RGB format
imageyuv = cv.cvtColor(imageyuv, cv.COLOR_YUV2RGB)
imagecv = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
# image processing
def showResult(label, imagecv=None, cmap=None):
    plt.figure(figsize=(8,8))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original')
    plt.xlabel('intensity value')
    plt.ylabel('intensity quantity')
    plt.subplot(1,2,2)
    plt.imshow(imagecv, cmap=cmap)
    plt.title(label)
    plt.axis('off')
    plt.show()

height,width =image.shape[:2]
def mean_filter(source, ksize):
    np_source = np.array(source)
    for i in range(height-ksize-1):
        for j in range(width-ksize-1):
           matrix = np.array(np_source[i:(i+ksize),j:(j+ksize)]).flatten() 
           mean = np.mean(matrix)
           np_source[i + ksize//2, j+ksize//2] = mean
    return np_source

def median_filter(source, ksize):
    np_source = np.array(source)
    for i in range(height-ksize-1):
        for j in range(width-ksize-1):
           matrix = np.array(np_source[i:(i+ksize),j:(j+ksize)]).flatten() 
           median = np.median(matrix)
           np_source[i + ksize//2, j+ksize//2] = median
    return np_source

def detecting_face(testPic):
    classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    train_path = 'trainingImage'

    tdir = os.listdir(train_path)

    face_list=[]
    class_list = []

    for idx, train_dir in enumerate(tdir):
        for image_path in os.listdir(f'{train_path}/{train_dir}'):
            path = f'{train_path}/{train_dir}/{image_path}'

            gray= cv.imread(path,0)
            faces = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
                
            if(len(faces)<1):
                continue
            for face_rec in faces:
                x,y,w,h = face_rec
                face_image = gray[y:y+w,x:x+h]
                face_list.append(face_image)
                class_list.append(idx)
    face_recog  = cv.face.LBPHFaceRecognizer_create()
    face_recog.train(face_list, np.array(class_list))
    full_path_list = []
    test_path = 'testImage'
    for path in os.listdir(test_path):
        full_path =f'{test_path}/{path}'
        full_path_list.append(full_path)
                
        if(len(faces)<1):
            continue
        
    # for a in full_path_list:
    #     print({a})
    if testPic==0:
        image = cv.imread(full_path_list[0])
        igray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(igray, scaleFactor=1.2, minNeighbors=5)
        for face_rec in faces:
                x,y,w,h = face_rec
                face_image = gray[y:y+w,x:x+h]
                res, conf =face_recog.predict(face_image)
                conf = math.floor(conf*100)/100
                cv.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 1)
                text = f'{tdir[res]} {str(conf)}%'
                cv.putText(image, text, (x,y-10),cv.FONT_HERSHEY_PLAIN, 1.5, (0,255,0),1)
                cv.imshow('result', image)
                cv.waitKey(0)
                
    elif testPic==1:
        image = cv.imread(full_path_list[1])
        igray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(igray, scaleFactor=1.2, minNeighbors=5)
        for face_rec in faces:
                x,y,w,h = face_rec
                face_image = gray[y:y+w,x:x+h]
                res, conf =face_recog.predict(face_image)
                conf = math.floor(conf*100)/100
                cv.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 1)
                text = f'{tdir[res]} {str(conf)}%'
                cv.putText(image, text, (x,y-10),cv.FONT_HERSHEY_PLAIN, 1.5, (0,255,0),1)
                cv.imshow('result', image)
                cv.waitKey(0)
            
    cv.destroyAllWindows()
curses.wrapper(printMenu, menuList,0)