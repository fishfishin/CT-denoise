"""""
Non locl means
"""

from PIL import  PSNR
import cv2
import numpy as np
from skimage import io,img_as_float,img_as_ubyte
from skimage.restoration import denoise_nl_means, estimate_sigma
from PIL import Image

img = io.imread("C:/Users/ZhenjuYin/Downloads/a1.tif", as_gray=False)
img = img[5,:,:]
img = img_as_float( img )
#img_name = "C:/Users/ZhenjuYin/Downloads/a1.tif" 
#img = Image.open(img_name)
#img = np.array(img)
sigma_est = np.mean(estimate_sigma(img,multichannel=False))

denoised_img =denoise_nl_means(img, h =1.15*sigma_est,fast_mode=True,patch_size=5,patch_distance=3,multichannel=False)
psnr = PSNR.PSNR(img, denoised_img)
print ("The PSNR between the two img after NLMF is %f" % psnr)
#img_as_8 = img_as_ubyte(denoised_img)
#final_img = cv2.cvtColor(img_as_8,cv2.COLOR_BGR2RGB)
cv2.imwrite('NLM.jpg',denoised_img)
#img_as_8 = img_as_ubyte(img)
#final_img = cv2.cvtColor(img_as_8,cv2.COLOR_BGR2RGB)
cv2.imwrite('NLM_o.jpg',img)