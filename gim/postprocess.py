import cv2 as cv
import numpy as np
import os

if __name__ == '__main__':
    path='output_school'
    outport='post_output_school'
    os.makedirs(outport, exist_ok=True)
    name=os.listdir(path)

    for i in range(len(name)):
        img=cv.imread(os.path.join(path,name[i]),cv.IMREAD_GRAYSCALE)
        img=cv.morphologyEx(img,cv.MORPH_OPEN,np.ones((10,10),np.uint8),iterations=5)
        img=cv.morphologyEx(img,cv.MORPH_CLOSE,np.ones((20,20),np.uint8),iterations=3)
        # cv.imshow('img',img)
        # key=cv.waitKey(0)
        # if key==27:
        #     break

        cv.imwrite(os.path.join(outport,name[i]),img)