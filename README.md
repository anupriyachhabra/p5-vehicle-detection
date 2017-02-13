# Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./test_images/test5.jpg
[image2]: ./output_images/s_channel.jpg
[image3]: ./output_images/hog_output.jpg
[image4]: ./output_images/first_pass/test1.jpg
[image5]: ./output_images/first_pass/test6.jpg
[image6]: ./output_images/final_output/test1.jpg
[image7]: ./output_images/heat_map/test1.jpg
[image8]: ./output_images/heat_map/test4.jpg
[image9]: ./output_images/final_output/test4.jpg


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

This file is the writeup for my project.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

First of all I read in all the car and non car images and made sure that they are an almost equivalent set and there is no bias
towards predicting cars or not cars. The code for this step is contained in line 177 through 190 of the file called `search_classify.py`.
I then called a function `extract_features` (lines 46 through 94 of `search_classify.py`) for both non-cars and cars to extract various
features of these images. The function `extract_features` contains a part to extract hog features in lines 48 through 60 of `search_classify.py`.
This file imports the `get_hog_features` helper function from file lesson_functions.py.).


#### 2. Explain how you settled on your final choice of HOG parameters.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).
I ran my initial code on all images in test_images folder and `test5.jpg` as shown below was the most problematic figure due to shadows.

![test5.jpg original][image1]

I calculated hog for the `test_images` to check the patterns that are created for these images.
I used the image `test5.jpg`. to further refine my code and choose parameters for `skimage.hog()`.
I finally settled in on s channel of HLS color space for this model as that was giving me best results for removing shadow effects.
Here is the s_channel of `test5.jpg`

![s channel][image2]

I then applied HOG with parameters of `orientations=7`, `pixels_per_cell=(16, 16)` and `cells_per_block=(4, 4)`
I chose these values as orientation 7 shows all features of a car and I doubled the values of `pixel_per_cell` and
`cell_per_block` as the images in this project are significantly bigger than those used in lessons.
I got the following output after running hog on `test5.jpg`
![hog on test5.jpg][image3]

After training the classifier as explained in next step I decided to use ALL the hog channels of HLS color space instead of just s channel


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I imported `LinearSVC` from `sklearn.svm` and used it to train a classifier. I have used all the images for cars and non cars
and split them into a training and test set using `sklearn.model_selection` `train_test_split` method . I put the data into
classifier using `LinearSVC.fit` method and then cross verified my results using `LinearSVC.score` method. I managed to get 94%
accuracy. The code for this is in line 231 to 249 of file `search_classify.py`

I then used all the hog channels for my image in `HLS` space and that improved the accuracy to 97% so I changed my code in hog to use
`ALL` color channels


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have implemented a sliding window search in line 143 through 150 in file `search_classify.py`. For this I have used a helper function
`slide_window` from file `lesson_functions.py`. This file `lesson_functions.py` contains lots of helper functions as given in Udacity
course material for this project.

I initially implemented my sliding window of size `96 X 96` with `50%` overlap in xand y direction.
I then got some false positives and so decided to increase the overlap to `70%` so that I have large overlapping windows with positive
searches so that I can easily remove the false positive using a heatmap method.

Further I was facing some issues in detecting cars closer to horizon so I added additional windows of size `48 X 48` with `50%` overlap.

I combined the 2 windows from above to define areas to search. This made my algorithm more robust.

Following are 2 images which are output of the applying the sliding_windows and searching.

![test1.jpg][image4]

![test6.jpg][image5]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

- To optimize my pipeline I undistorted the initial image using the calibration matrix saved from Project 4. This made my pipeline
detect more positives in the sides of the images.
- Also as explained earlier I used all the channels of hog to get accuracy from 94% to 97%.
- When drawing the bounding boxes over my image I have ignored the bounding boxes where minimum value of x is too small as that detects
cars in opposite direction.
- I have changed the y_start and y_stop to not include the portion of image too close to horizon and too close to the car. This helped me to
get rid of lots of false positives like trees and road signs.

Following is an example of the final output of my pipeline.
![final output test1.jpg][image6]


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a
heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify
individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.
I constructed bounding boxes to cover the area of each blob detected. Also when drawing the bounding boxes over my image I
have ignored the bounding boxes where minimum value of x is too small as that detects cars in opposite direction.

The code for implementing above is in line 97 to 127 of file `search_classify.py` and is called from function `pipeline` of same file.


### Here are 2 images from test_images folder with heatmaps
|Heatmap of test1.jpg | Heatmap of test4.jpg|
|---------------------|---------------------|
|![heatmap test1.jpg][image7]|![heatmap test4.jpg][image8]|


### Here the resulting bounding boxes on the images
![bounding box test1.jpg][image6]
![bounding box test4.jpg][image9]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- My pipeline detects car in opposite direction, for that I had to remove certain area of image from left side.
The pipeline might fail if car is not in the left most lane as it might falsely remove some cars in the lanes left of it.
- To make the pipeline more robust I would use something to average out the heatmaps across frames to removed the false positives in a
better way.
- This pipeline will only work on images of size `1280 X 720`. It will fail on images of any other size. I would like to make it more
robust so that it works on any image size.


