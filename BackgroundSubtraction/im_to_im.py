import cv2
import numpy as np
import glob
img_array = []
for filename in glob.glob('DATA/baseline/baseline/office/input/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)




for i in range(len(img_array)):
    filepath = 'DATA/baseline/results/office/'
    filename = 'out' + str(i).zfill(6)+'.jpg'
    cv2.imwrite(filepath + filename, img_array[i])

"""
Pasos para implementar el exponential filter:
1.Construir el modelo de background a partir de los primeros n frames
    1.1.
    
"""