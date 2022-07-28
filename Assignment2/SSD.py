import numpy as np
from PIL import Image

def SSD(leftImage, rightImage, windowSize, DSR):
    #load images
    leftImage = Image.open(leftImage).convert('L')
    leftImageArray = np.asarray(leftImage)
    rightImage = Image.open(rightImage).convert('L')
    rightImageArray = np.asarray(rightImage)
    width, height = leftImage.size
    #init disparity map
    disparityMap = np.zeros((width, height), np.uint8)
    disparityMap.shape = height, width
    window = int(windowSize / 2)
    #loop
    for y in range(window, height - window):
        print("\rProcessing.. %d%% complete" % (y / (height - window) * 100), end="", flush=True)
        for x in range(window, width - window):
            best = 0
            prevSSD = 65534
            for dsr in range(DSR):
                ssd = 0
                temp = 0
                for v in range(-window, window):
                    for u in range(-window, window):
                        temp = int(leftImageArray[y + v, x + u]) - int(rightImageArray[y + v, (x + u) - dsr])
                        ssd += temp * temp
                if ssd < prevSSD:
                    prevSSD = ssd
                    best = dsr
            disparityMap[y, x] = best * (255 / DSR)  
    Image.fromarray(disparityMap).save('pred/Art/disp1.png')
    #Image.fromarray(disparityMap).save('pred/Dolls/disp1.png')
    #Image.fromarray(disparityMap).save('pred/Reindeer/disp1.png')

if __name__ == '__main__':
   SSD("gt/Art/view1.png", "gt/Art/view5.png", 6, 200)
   #SSD("gt/Dolls/view1.png", "gt/Dolls/view5.png", 6, 200)
   #SSD("gt/Reindeer/view1.png", "gt/Reindeer/view5.png", 6, 200)
