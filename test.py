import cv2
import numpy as np
import matplotlib.image as mpimg


image = mpimg.imread("test_images/test5.jpg")
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
#s_channel = image[:,:,2]
cv2.imwrite('test_output.jpg', hls);
