import cv2
import numpy as np

DATASETS = [ 'DUTS','PASCAL-S', 'ECSSD', 'HKU-IS','DUT-OMRON']
for e in DATASETS:

    with open('./data/'+e+'/test.txt', 'r') as lines:
        
        for line in lines:
                    
            maskpath  = './data/'+e+'/mask/' + line.strip() + '.png'
            img = cv2.imread(maskpath,cv2.IMREAD_GRAYSCALE)
            img_shape = img.shape
            img[img<128]=0
            img[img>128]=255
            img_edge=np.zeros(img_shape)

            for i in range(img_shape[0]):
                for j in range(img_shape[1]):
                    try:
                        if img[i][j] == 0:
                            if img[i+1][j]==255 or img[i-1][j]==255 or img[i][j+1]==255 or img[i][j-1]==255 or img[i-1][j-1]==255 or img[i+1][j+1]==255 or img[i-1][j+1]==255 or img[i+1][j-1]==255:
                                img_edge[i][j]=255
                        if img[i][j] == 255:
                            if img[i+1][j]==0 or img[i-1][j]==0 or img[i][j+1]==0 or img[i][j-1]==0 or img[i-1][j-1]==0 or img[i+1][j+1]==0 or img[i-1][j+1]==0 or img[i+1][j-1]==0:
                                img_edge[i][j]=255
                    except IndexError:
                        continue

            cv2.imwrite('./data/edgemask/'+ e+'/mask/'+line.strip() + '.png',img_edge)
