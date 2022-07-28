import cv2
from math import floor
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def calhist(image):
    height, weight, color = image.shape
    colorHist = np.zeros([16 * 16 * 16, 1], np.float32)
    size = 256 / 16
    for row in range(height):
        for col in range(weight):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = int(b / size) * 16 * 16 + int(g / size) * 16 + int(r / size)
            colorHist[int(index), 0] += 1
    return colorHist

def hist_compare(image1, image2):
    histogram1 = calhist(image1)
    histogram2 = calhist(image2)
    sim = cv2.compareHist(histogram1, histogram2, cv2.HISTCMP_CORREL)
    return sim
    #print(sim)


if __name__ == '__main__':
    img_query_path="{}/{}.jpg".format("./datasets/query", {})
    img_gallery_path="{}/{}.jpg".format("./datasets/gallery", {})
    img_query=[]
    img_gallery=[]
    img_sim={}
    for i in range(49):
        img_query.append(img_query_path.format(i))
        img = Image.open(img_query[i])
        img_1 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        for j in range(28492):
            img_gallery.append(img_gallery_path.format(j))
            img_2= Image.open(img_gallery[j])
            img_3 = cv2.cvtColor(np.asarray(img_2), cv2.COLOR_RGB2BGR)
            result = hist_compare(img_1,img_3)
            img_sim.update({j:result})
        value = sorted(img_sim.items(), key=lambda e: e[1],reverse=True)
        value_1 = []
        for k in value:
            value_1.append(k[0])
        file = open('rankList1.txt', 'a')
        file.write('Q{}:'.format(i+1))
        for ip in value_1:
            file.write(str(ip))
            file.write(' ')
        print(value_1)
        #print(img_sim)
        img_sim.clear()
