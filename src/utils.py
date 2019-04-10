import cv2 as cv
import numpy as np
def cv_imread(file_path):
    cv_img = cv.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img