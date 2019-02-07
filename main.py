import os
import math
import pickle

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import data, util
from skimage.measure import label, regionprops
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression


N_ROW = 900
N_COL = 1200
PLT_COL = 4
PLT_ROW = 1


def score_samples(test_X, mus, covs, pis):
    def __scale(pixels):
        return np.array(pixels / 255.0)

    def __prior_probability(X, N):
        prob_x_n = np.zeros((len(X), N))
        for n in range(N):
            prob_x_n[:, n] = multivariate_normal.pdf(X, mean=mus[n], cov=covs[n])
        
        return prob_x_n
    
    test_X = __scale(test_X)
    prob_x_n = __prior_probability(test_X, len(mus))
    return np.log(np.sum(np.exp(np.log(prob_x_n) + np.log(pis)), axis=1)) 


def segmentation(image, mus, covs, pis):
    THRESH = 4.5
    res = score_samples(np.reshape(image, (N_ROW * N_COL, 3)), mus, covs, pis)
    return np.reshape([True if x > THRESH else False for x in res], (N_ROW, N_COL))


def label_region(seg_img):
    BARREL_RATIO_MAX = 3.5
    BARREL_RATIO_MIN = 1.01
    AREA_MIN = 1000
    AREA_RATIO_MIN = 0.3
    NUM_BARRELS_PICKING = 2

    img = util.img_as_ubyte(seg_img)
    labeled = label(img, connectivity=img.ndim) 
    
    rects = []
    for region in regionprops(labeled):
        if region.area >= AREA_MIN and float(region.area) / region.convex_area > AREA_RATIO_MIN:
            minr, minc, maxr, maxc = region.bbox
            h, w = float(maxr - minr), float(maxc - minc)
            ratio = max(h, w) / min(h, w)
            if ratio < BARREL_RATIO_MIN or ratio > BARREL_RATIO_MAX:
                continue
            
            rects.append(((minc, minr), w, h, region))
            
    if len(rects) == 0:
        return rects

    rects.sort(key=lambda x: x[3].area)
    return rects[-NUM_BARRELS_PICKING:]



def main():

    with open('./models/linear.pkl', 'rb') as handle:
        distance_model = pickle.load(handle)

    gmm_mus = np.load('./models/GMM_mus.npy')
    gmm_covs = np.load('./models/GMM_covs.npy')
    gmm_pis = np.load('./models/GMM_pis.npy')


    folder = "Test_Set"

    results = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_YCR_CB = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)        

        plt.figure(figsize=[10, 5])

        plt.subplot(PLT_ROW, PLT_COL, 1)
        plt.imshow(img_RGB)
        plt.axis("off")
        plt.title('RGB')

        plt.subplot(PLT_ROW, PLT_COL, 2)
        plt.imshow(img_YCR_CB)
        plt.axis("off")
        plt.title("YCR_CB")

        img_seg = segmentation(img_YCR_CB, gmm_mus, gmm_covs, gmm_pis)
        plt.subplot(PLT_ROW, PLT_COL, 3)
        plt.imshow(img_seg)
        plt.axis("off")
        plt.title("Segmentation")


        axes = plt.subplot(PLT_ROW, PLT_COL, 4)
        plt.imshow(img_seg)
        plt.axis("off")
        
        regions = label_region(img_seg)
        if len(regions) > 0:
            X = []
            for region in regions:
                X.append([region[3].centroid[0], 
                          region[3].centroid[1],
                          region[1],
                          region[2],
                          region[3].orientation])


            dists = np.absolute(distance_model.predict(X))

            for i in range(len(regions)):
                rect = regions[i]
                axes.add_patch(mpatches.Rectangle(rect[0], rect[1], rect[2], fill=False, edgecolor='red', linewidth=2))
                print('ImageNo = {}, CentroidX = {:.2f}, CentroidY = {:.2f}, Distance = {:.2f}'.format(filename, rect[3].centroid[0], rect[3].centroid[1], dists[i]))


            dist = "{0:.2f}".format(dists[0])
            if len(dists) > 1:
                dist += "_" + "{0:.2f}".format(dists[1])
            plt.title("Distance: " + dist)

        else:
            plt.title("Distance: N/A")


        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()