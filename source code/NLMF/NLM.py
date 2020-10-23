"""""
Non locl means
"""
import skimage
from PIL import  PSNR
import cv2
import numpy as np
from skimage import io,img_as_float,img_as_ubyte
from skimage.restoration import denoise_nl_means, estimate_sigma
from PIL import Image
import matplotlib.pyplot as plt

def NLM(img):
    
    sigma_est = np.mean(estimate_sigma(img,multichannel=False))

    Final_img =denoise_nl_means(img, h =1.15*sigma_est,fast_mode=True,patch_size=5,patch_distance=3,multichannel=False)
    psnr = PSNR.PSNR(img, Final_img)
    print ("The PSNR between the two img after NLMF is %f" % psnr)
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('NLM')
    axs[ 0].imshow(img, cmap='gray')
    axs[ 1].imshow(Final_img, cmap='gray')
    axs[ 0].set_title('original')
    axs[ 1].set_title('final')

    plt.show()

    return Final_img



if __name__ == '__main__':
    image = skimage.external.tifffile.imread("C:/Users/ZhenjuYin/Downloads/a1.tif")

    
    for i in range(16):
        img = image[i,:,:]
        image[i,:,:] = NLM(img)
    skimage.external.tifffile.imsave('t.tif', image)
    image = skimage.external.tifffile.imread("C:/Users/ZhenjuYin/Downloads/t.tif")
    skimage.external.tifffile.imshow(image[15,:,:],photometric='miniswhite' )
