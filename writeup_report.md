## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hogfeat]: ./output_images/HOG_features.png
[slidingwnd]: ./output_images/sliding_wnd.png
[unfiltered]: ./output_images/unfiltered_rects.png
[heatmap]: ./output_images/heatmap.png
[image_pipeline]: ./output_images/image_pipeline.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in code cells 2-9 of the IPython notebook.

I started by creating lists of `vehicle` and `non-vehicle` image classes. Then I define a few constants, which are used throughout the notebook, which specify the features being used (explained in the next section). I downsample the image to the size 24x24 (rationale follows) prior to extracting features. Now that the feature vector structure is defined, I extract feature vectors from all positive and negative samples, perform normalization using `StandardScaler`, and also split everything into the training and test sets.

#### 2. Explain how you settled on your final choice of HOG parameters.

The constants, defining the feature vector, were chosen as follows:

* Number of HOG orientation bins value of 8, as it corresponds to 45 degree increments;
* Number of cells per block is 1, as in my experiments with this dataset, any other number grew the feature vector quadratically, without much benefits to the accuracy of the classifier on the test set, but with severe run time impacts;
* Number of pixels per HOG cell side defaults to 8;
* The classifier size was chosen to be 24x24 window, meaning it consists of 3x3 HOG cells/blocks, provided the above choices;
* HOG sliding window was chosen as 1, because it is already "too much" compared to the window size, meaning 33% increments.

The choice of classifier window (24x24 instead of 64x64) is reasoned by the tradeoff in runtime performance vs accuracy of the classifier. Apparently, for the task of rough bounding box detection, the selected size allows to capture car outlines better, than with 64x64 window, due to the fact, that the gradients of a smaller image, capture more spatial information, than of the larger image (concentrate on details). This also allowed to drop the block of cells size to one, cause larger blocks of cells work better when the original gradients don't capture enough space.

I experimented with all features, described in the course materials, but chose to use only HOG:
* Color spatial binning is unreliable and leads straight to overfitting, due to major color/brightness variation across class samples;
* Color histograms are better, but still, not adding up to accuracy
* HOG features give the largest contribution as a reliable discriminator, so I kept only them.

The only use of color made in this project is related to HOG features: I convert the input RGB windows to YUV color space (YUV), calculate HOG on each channel, and concatenate outputs into a single feature vector.

The resulting feature vector is quite compact, which makes it easier to use in real time processing (although this notebook is by no means close to realtime due to many other optimizations missing).

FeatureVecLen = (CELLS=9) * (BINS=8) * (COLORPLANES=3) = 216 floats

Below is an example of one sample and its HOG visualization:
![][hogfeat]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training the classifier is in cells 12-13. As can be seen in cell 12, I experimented with the GridSearchCV instance to find the optimal parameters. Curiously enough, the default parameters of kernel='rbf', C=1.0, gamma='auto' (which corresponds to the inverse of feature vector length) give the best accuracy and take least time to train.

Evaluation of the classifier trained with the default parameters, provided the above feature vector structure, yields accuracy of 0.9904, which is an acceptable result for this task.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used multi-scale exhaustive search with ROI constraints. The idea is to look for the small objects around horizon, and larger objects anywhere around the car hood. For performance considerations, it is important to narrow down the smallest scale area of search.

Here is the visualization of my scales, evaluated on a sequence of images with a car at different distances. The red rectangle is the search area for a particular scale. This approach also helps to prevent false positives around cluttered areas like treetops, where cars are not expected.

![][slidingwnd]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

With all scales combined, here is an example of the detector working on standalone images:
![][unfiltered]

After collecting all detections, a heatmap is created to facilitate futher bounding boxes grouping. An alternative approach would be to use `cv2.groupRectangles`, which should be less computationally expensive, but doesn't permit creating a heatmap.
![][heatmap]

Finally, all steps are combined: heatmap is placed on top of the image, grouped detections are highlighted with rectangles:
![][image_pipeline]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a visualization (youtube) of the pipeline steps applied to the project video (click the image):

[![](https://img.youtube.com/vi/NHWOLMmx2nU/0.jpg)](https://www.youtube.com/watch?v=NHWOLMmx2nU)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

First level of filtering is thresholding the heatmap - anything below the contribution of one pixel is rejected, which helps to fight sporadic false detections.

Subsequently, I apply rolling update to the heatmap accumulator buffer, which is fused with the heatmap found during the next step. This helps to reduce jerky updates and smooths detections.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* Video pipeline asks for a proper tracking algorithm, so there is no need to perform full search every frame. This could bump up runtime performance by a lot.
* Performance of 720p processing with all visualizations disabled is around 10 fps with the laptop CPU, which is not nearly sufficient for a realtime requirement. A proper GPU optimization could help to address performance issues.
* Speaking of traditional computer vision approaches, a cascade of classifiers could help to reduce execution time.
* The current pipeline is tweaked to perform on evaluation videos. Any uphill/downhill road will constitute a problem for the pipeline and require either retuning or increasing the search areas.
* It would be interesting to use multiple LODs (levels of details) with the HOG features, so the final feature vector captures gradients at different LODs.
