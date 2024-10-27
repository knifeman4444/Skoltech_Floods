import os
import cv2
from PIL import Image


for i in range(1, 5):
    path = f'../dataset/test/images/dwm_{i}.png'
    pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    threshold = int(0.9 * 255)

    pic[pic < threshold] = 0
    pic[pic >= threshold] = 1

    out_path = f'../dataset/result/images/{i}.tif'

    im = Image.fromarray(pic)
    im.save(out_path)
