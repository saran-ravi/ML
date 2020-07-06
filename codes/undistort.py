import numpy as np
import cv2
import glob

nffile = np.load('Camera.npz')

mtx = nffile['mtx']

dist =  nffile['dist']




#cap = cv2.VideoCapture(0)
#ret,imgl = cap.read()
imgl=cv2.imread('l.jpg')
#imgr= cv2.imread('r.jpg')
h,  w = imgl.shape[:2]
cv2.imshow("Original",imgl)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0.5)
print(w,h)
print(roi)
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(imgl,mapx,mapy,cv2.INTER_LINEAR)
# undistort
dst = cv2.undistort(imgl, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst1 = dst[y:y+h, x:x+w]
#cv2.imwrite('right1.png', dst1)
print(dst)
cv2.imwrite('undist_left.png', dst1)
cv2.imshow('undist_left',dst1)
cv2.imshow('dist_left',dst)

cv2.waitKey()
cap.release()
cv2.destroyAllWindows()
