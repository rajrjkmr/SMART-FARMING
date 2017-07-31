import socket
import Image
import math
import os
import operator
import cv2
import numpy as np
from matplotlib import pyplot as plt

def camera():
	cap = cv2.VideoCapture(0)
	s,photo=cap.read()
	cv2.imwrite('4.jpg',photo)
	image_similarity_histogram_via_pil()
def get_thumbnail(image, size=(128,128), stretch_to_fit=False, greyscale=False):
    
    if not stretch_to_fit:
        image.thumbnail(size, Image.ANTIALIAS)
    else:
        image = image.resize(size); # for faster computation
    if greyscale:
        image = image.convert("L")  # Convert it to grayscale.
    return image
def match():

	img_rgb = cv2.imread('4.jpg')
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	template = cv2.imread('1.jpg',0)
	w, h = template.shape[::-1]
	res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
	threshold = 0.1
	print res	
	if res[0][0]>= threshold:
	   os.system('/home/raj/tg/bin/telegram-cli -k server.pub -W -e "msg Phone_number As tha Plants are Alive.I have Switched ON the Motor..."')
	   os.system('/home/raj/tg/bin/telegram-cli -k server.pub -W -e "send_photo Phone_number 4.jpg"')
	   print "match"
	

def image_similarity_histogram_via_pil():
    image1 =Image.open('4.jpg')
    image2 =Image.open('1.jpg')
    image1 = get_thumbnail(image1)
    image2 = get_thumbnail(image2)
    h1 = image1.histogram()
    h2 = image2.histogram()
    rms = math.sqrt(reduce(operator.add,  list(map(lambda a,b: (a-b)**2, h1, h2)))/len(h1) )
    print rms
    if rms <= 100:
       match()
    
def face():
	face_cascade = cv2.CascadeClassifier('/root/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('/root/opencv/data/haarcascades/haarcascade_eye.xml')
	img = cv2.imread('4.jpg')

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	res = cv2.resize(img,(700,700), interpolation = cv2.INTER_CUBIC)
	cv2.imshow('img',res)
	cv2.waitKey(0)





host = '192.168.43.201'
port = 82
address = (host, port)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(address)
server_socket.listen(5)
print "Listening for client . . ."
conn, address = server_socket.accept()
while True:
   	output = conn.recv(2048);
   	print output
   	data=float(output)
	if data==2047:
		camera()

 	 	