import cv2 as cv
import numpy as np
import src.config
from src.utils import cv_imread
import random
base_dir = src.config.get_RES_DIR()

def get_components(binary,image_name):
    connectivity = 8
    num_comps, comps, stats, centroids = cv.connectedComponentsWithStats(binary,connectivity,cv.CV_32S)
    print(comps.shape)
    print(type(stats))
    colors = [(0,0,0)]*num_comps
    for i in range(1,num_comps):
        colors[i] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    height, width = binary.shape
    colored_map = np.zeros((height,width,3), np.uint8)
    for x in range(width):
        for y in range(height):
            label = comps[x,y]
            colored_map[x][y] = colors[label]
    cv.imshow('comps',colored_map)
    cv.waitKey(0)



if __name__ == '__main__':
    file_name = r'\1_pre'
    posfix = r'.png'
    file_path = base_dir+file_name+posfix
    file_path = r"{}".format(file_path)
    src = cv_imread(file_path)
    cv.imshow('src',src)
    cv.waitKey(0)
    get_components(src,'1')
    # cv.imshow('preprocessed',dst)

    cv.destroyAllWindows()


