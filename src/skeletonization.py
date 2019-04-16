import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.filters import threshold_otsu
from skimage import color
import cv2 as cv
import numpy as np

def neighbours(x, y, image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],  # P2,P3,P4,P5
            img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]  # P6,P7,P8,P9


def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]  # P2, P3, ... , P8, P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)


def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1  # the points to be removed (set as 0)
    while changing1 or changing2:  # iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape  # x for rows, y for columns
        for x in range(1, rows - 1):  # No. of  rows
            for y in range(1, columns - 1):  # No. of columns
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and  # Condition 0: Point P1 in the object regions
                        2 <= sum(n) <= 6 and  # Condition 1: 2<= N(P1) <= 6
                        transitions(n) == 1 and  # Condition 2: S(P1)=1
                        P2 * P4 * P6 == 0 and  # Condition 3
                        P4 * P6 * P8 == 0):  # Condition 4
                    changing1.append((x, y))
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and  # Condition 0
                        2 <= sum(n) <= 6 and  # Condition 1
                        transitions(n) == 1 and  # Condition 2
                        P2 * P4 * P8 == 0 and  # Condition 3
                        P2 * P6 * P8 == 0):  # Condition 4
                    changing2.append((x, y))
        for x, y in changing2:
            Image_Thinned[x][y] = 0
    return Image_Thinned


def fix_skeleton(image):
    image = image.astype(np.uint8)
    # gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    kernal = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    fixed = cv.morphologyEx(image,cv.MORPH_DILATE,kernal,iterations=1)
    return fixed

if __name__ == '__main__':
    "load image data"
    file_path = r'C:\Users\saiju\PycharmProjects\Chinese_Brush_Word_Recognition\resources\4.jpg'
    Img_Original = color.rgb2gray(io.imread(file_path))  # Gray image, rgb images need pre-conversion

    "Convert gray images to binary images using Otsu's method"

    Otsu_Threshold = threshold_otsu(Img_Original)
    BW_Original = Img_Original < Otsu_Threshold  # must set object region as 1, background region as 0 !
    "Apply the algorithm on images"
    # BW_Original = cv.GaussianBlur(BW_Original.astype(np.uint8),3,0)
    BW_Skeleton = zhangSuen(BW_Original)
    print(type(BW_Skeleton))
    fixed = fix_skeleton(BW_Skeleton)
    # BW_Skeleton = BW_Original
    "Display the results"
    fig, ax = plt.subplots(1, 3)
    ax1, ax2 , ax3 = ax.ravel()
    ax1.imshow(BW_Original, cmap=plt.cm.gray)
    ax1.set_title('Original binary image')
    ax1.axis('off')
    ax2.imshow(BW_Skeleton, cmap=plt.cm.gray)
    ax2.set_title('Skeleton of the image')
    ax2.axis('off')
    ax3.imshow(fixed, cmap=plt.cm.gray)
    ax3.set_title('fixed Skeleton of the image')
    ax3.axis('off')
    plt.show()

