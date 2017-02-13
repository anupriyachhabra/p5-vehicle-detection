import cv2
import numpy as np
import matplotlib.image as mpimg
from skimage.feature import hog


image = mpimg.imread("test_images/test5.jpg")
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
s_channel = image[:,:,2]
cv2.imwrite('output_images/s_channel.jpg', s_channel);
cv2.imwrite('output_images/test_output.jpg', hls);

features, hog_image = hog(s_channel, orientations=7,
                                  pixels_per_cell=(16, 16),
                                  cells_per_block=(4, 4),
                                  transform_sqrt=False,
                                  visualise=True, feature_vector= True)

cv2.imwrite('output_images/hog_output.jpg', hog_image);
