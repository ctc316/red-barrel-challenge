import os
import sys

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
from roipoly import RoiPoly


def main():
    dir_train = "./2019Proj1_train/"
    dir_labeled_train = "./labeled_train/"

    if not os.path.exists(dir_labeled_train):
        os.makedirs(dir_labeled_train)

    filelist = os.listdir(dir_train)
    num_file = len(filelist)

    startIdx = 0
    endIdx = num_file - 1
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        startIdx = int(sys.argv[1])

    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        endIdx = int(sys.argv[2])

    print(endIdx)
    for i in range(max(0, startIdx), min(num_file - 1, endIdx) + 1):
        filename = filelist[i]
        name, extension = os.path.splitext(filename)
        image = Image.open(dir_train + filename)
    
        plt.title("Labeling: " + str(i) + ". " + name)
        plt.imshow(image)
        roi = RoiPoly(color='r')
        
        # show result
        plt.title("Result: " + str(i) + ". " + name)
        plt.imshow(image)
        roi.display_roi()
        plt.show()

        # get mask
        imgSize = np.shape(image)
        mask = roi.get_mask(np.zeros(imgSize[0:2], dtype=bool))
        
        # show mask
        # plt.title("Mask: " + str(i) + ". " + name)
        # plt.imshow(mask)
        # plt.show()

        np.save(os.path.join(dir_labeled_train, name + '.npy'), mask)
        print("Finish labeling: " + dir_labeled_train + name + '.npy')
        plt.close()



if __name__ == '__main__':
    main()