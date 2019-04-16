import cv2 as cv

if __name__ == '__main__':
    file_path = r'C:\Users\saiju\PycharmProjects\Chinese_Brush_Word_Recognition\resources\3.png'
    image = cv.imread(file_path,-1)
    alpha = image[:,:,3]#extract it
    binary = ~alpha #invert b/w
    cv.imshow('src',binary)
    cv.waitKey(0)
    cv.destroyAllWindows()