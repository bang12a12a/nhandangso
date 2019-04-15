import cv2
import numpy as np

img = cv2.imread('digits.png',0)
cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

x = np.array(cells)
train = x[:,:50].reshape(-1,400).astype(np.float32)
k= np.arange(10)
train_labels = np.repeat(k, 250)[:,np.newaxis]
knn = cv2.ml.KNearest_create()
knn.train(train, 0 ,train_labels)

im= cv2.imread('test02.png')
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
for rect in rects:
    
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    roi = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))   
    x2 = np.array(roi)
    test2 = x2.reshape(-1,400).astype(np.float32)
    k1,k2,k3,k4 = knn.findNearest(test2,5)
    cv2.putText(im, str(int(k1)), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 250, 255), 2)
cv2.imshow("Ket Qua Nhan Dang", im)

cv2.waitKey()

    


