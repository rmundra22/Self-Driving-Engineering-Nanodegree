# **Finding Lane Lines on the Road** 

> Detecting Lane Lines on the Road using computer vision. The goals / steps of this project are the following:


* To detect the line segments in the image.
* Then average/extrapolate them and draw them onto the image for display.
* Once we have a working pipeline, try it out on the video stream below.
* Trying our lane finding pipeline on the challenge video. Does it still work? To figure out a way to make it more robust?


### 1. Overview of the pipeline

> The submission file is the jupyter notebook P1.ipynb where the processing pipeline is implemented. The pipeline consists on six steps represented by nine different functions:


* **grayscale:** Applies the Grayscale transformation and return a gray scaled version of the input image using cv2.cvtColor method.

* **gaussian_blur:** Applies a Gaussian Noise Kernel to blur or smooth out the provided image using cv2.GaussianBlur method.

* **canny:** Applies a Canny transformation to find edges on the image using cv2.Canny method.

* **region_of_interest:** Applies an image mask. Only keeps the region of the image defined by the polygon formed from `vertices`. The rest of the image is set to black. Eliminate parts of the image that are not interesting in regards to the line detection.

* **hough_lines:** Applies a Hough transformation on canny transformed images to find the lines using cv2.HoughLinesP. Checks for collinearity of the points detected as an edge. Further helps in separating left lane from the right lane using draw|_lines function.

* **draw_lines:** Applies a method to separate points of left lane from points on right lane line. Further average/extrapolate detected line segments to map out the full extent of the lane. This function draws `lines` with `color` and `thickness`. Lines are drawn on the image inplace (mutates the image).

* **extrapolate_lines:** Applies a method for extrapolating lane lines to perfect ego-vehicle manoeuvre across roads. Checks for the degree of the polynomial that will fit the points detected in right/left lanes. (Helper fucntions: linear, cubic, quadratic, select_degree to add fucntionality to this method)

* **weighted_img:** Merges the output of houghAction with the original image to represent the lines on it, even in an semi-transparent manner.

* **process_image:** Applies a method for lane line detection from coloured image input.

* **process_clip:** Applies on series of images for lane line detection and outputs a video clip with detected lane lines.


> NOTE: First, the pipeline is tested agains the images contained at test_images. The output of each step is saved in a directory:

    test_images_gray
    test_images_blur
    test_images_canny
    test_images_region
    test_images_hough
    test_images_merged


> After that test, the pipeline is consolidated on a single function process_clip to apply it on a video frame. The video after the transformation are saved on the [test_videos_output] directory.


> Trying our lane finding pipeline on the challenge.mp4. To figure out a way to make it more robust. The video after the transformation are saved on the [test_videos_output] directory.


### 2. Potential shortcomings with our current pipeline


* Lanes can be detected only in urban environment where we can find perfect side lanes on roads. Difficult to detect lanes in case of semi-urban or rural areas.
* Lane lines can be detected only on highways with lesser traffic. It will fail to detect lanes in case of heavy traffic situations.
* Lane detection may even fail with bad weather conditions like heavy snow-fall etc. Also if parts of lanes are damaged then it will fail there too.


### 3. Suggested possible improvements to our current pipeline

* Adding functionality to detect lanes in medium range traffic by some interpolating/ extrapolating detected lane lines across the video timeframes.
* Improving the detection of lane lines with complex curvatures by addding a functionality to fit higher degree polynomial curves after detecting points in the right/left lanes.

