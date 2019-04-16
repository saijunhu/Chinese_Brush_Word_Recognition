import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from src.utils import cv_imread
import src.config

base_dir = src.config.get_RES_DIR()

def preprocess(image, image_name):
    print(image.shape)
    gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    blured = cv.bilateralFilter(gray,5,25,25)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    openout = cv.morphologyEx(blured,cv.MORPH_OPEN,kernel,iterations=2)
    closeout = cv.morphologyEx(openout,cv.MORPH_OPEN,kernel,iterations=2)
    ret, binary = cv.threshold(closeout,0,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
    cv.imshow('binary',binary)
    cv.imwrite(base_dir+image_name+'_pre.png',binary)
    return binary


if __name__ == '__main__':
    file_name = r'\1'
    posfix = '.png'
    file_path = base_dir+file_name+posfix
    file_path = r"{}".format(file_path)
    src = cv_imread(file_path)
    cv.imshow('src',src)
    cv.waitKey(0)
    dst = preprocess(src,file_name)
    cv.imshow('preprocessed',dst)
    cv.waitKey(0)
    cv.destroyAllWindows()
