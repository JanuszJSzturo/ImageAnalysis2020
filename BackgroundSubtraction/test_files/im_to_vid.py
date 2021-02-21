import cv2
import numpy as np
import glob
img_array = []
for filename in glob.glob('DATA/baseline/results/highway_MOG/*jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('resultTest.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, size)


for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

"""
Pasos para implementar el exponential filter:
1.Construir el modelo de background a partir de los primeros n frames
    1.1.
    
"""