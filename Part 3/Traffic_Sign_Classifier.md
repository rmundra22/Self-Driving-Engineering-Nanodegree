
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

## TODO:

- [x] completed all of the code implementations
- [x] code cells need to have been run so that reviewers can see the final implementation and output
- [x] finalize your work by exporting the iPython Notebook as an HTML document
- [x] a writeup to complete either a markdown file or a pdf document following the Rubric points
- [x] adding requirements for enhancing the project beyond the minimum requirements
- [x] discuss the additional results in the writeup file


## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

- [x] Complete the basic data summary

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


    Number of training examples = 34799
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43


### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

- [x] Basic exploratory and visulizations of the dataset
- [x] the distribution of classes in the training, validation and test set **Distribution is same for most classes**

### A. Dataset Visualization
![png](output_14_0.png)


### B. Exploratory Analysis
![png](output_17_1.png)
![png](output_18_1.png)
![png](output_19_1.png)


> Looking at the distributions graphs above for training, validation and testing - they seem to be almost similar. hence we can conclude that validation and testing set are a good representative (proxy) of the training dataset.

----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- [x] Neural network architecture (is the network over or underfitting?)
- [x] Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- [x] Number of examples per label (some have more than others).
- [x] Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### A. Preprocessing RGB images

- Converting RGB images to Gray images
- Converting Gray images to Scaled / Normalized images
- Converting Scaled Images to Contrast Stretched Images

#### Note
> Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
> Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
> Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

![png](output_25_0.png)
![png](output_26_0.png)
![png](output_27_0.png)


### B. Reading through Labels of training dataset

# print statistic info

    No. of Images in class  0  : 180.0 	 Traffic Signal Label : Speed limit (20km/h)
    No. of Images in class  1  : 1980.0 	 Traffic Signal Label : Speed limit (30km/h)
    No. of Images in class  2  : 2010.0 	 Traffic Signal Label : Speed limit (50km/h)
    No. of Images in class  3  : 1260.0 	 Traffic Signal Label : Speed limit (60km/h)
    No. of Images in class  4  : 1770.0 	 Traffic Signal Label : Speed limit (70km/h)
    No. of Images in class  5  : 1650.0 	 Traffic Signal Label : Speed limit (80km/h)
    No. of Images in class  6  : 360.0 	 Traffic Signal Label : End of speed limit (80km/h)
    No. of Images in class  7  : 1290.0 	 Traffic Signal Label : Speed limit (100km/h)
    No. of Images in class  8  : 1260.0 	 Traffic Signal Label : Speed limit (120km/h)
    No. of Images in class  9  : 1320.0 	 Traffic Signal Label : No passing
    No. of Images in class 10  : 1800.0 	 Traffic Signal Label : No passing for vehicles over 3.5 metric tons
    No. of Images in class 11  : 1170.0 	 Traffic Signal Label : Right-of-way at the next intersection
    No. of Images in class 12  : 1890.0 	 Traffic Signal Label : Priority road
    No. of Images in class 13  : 1920.0 	 Traffic Signal Label : Yield
    No. of Images in class 14  : 690.0 	 Traffic Signal Label : Stop
    No. of Images in class 15  : 540.0 	 Traffic Signal Label : No vehicles
    No. of Images in class 16  : 360.0 	 Traffic Signal Label : Vehicles over 3.5 metric tons prohibited
    No. of Images in class 17  : 990.0 	 Traffic Signal Label : No entry
    No. of Images in class 18  : 1080.0 	 Traffic Signal Label : General caution
    No. of Images in class 19  : 180.0 	 Traffic Signal Label : Dangerous curve to the left
    No. of Images in class 20  : 300.0 	 Traffic Signal Label : Dangerous curve to the right
    No. of Images in class 21  : 270.0 	 Traffic Signal Label : Double curve
    No. of Images in class 22  : 330.0 	 Traffic Signal Label : Bumpy road
    No. of Images in class 23  : 450.0 	 Traffic Signal Label : Slippery road
    No. of Images in class 24  : 240.0 	 Traffic Signal Label : Road narrows on the right
    No. of Images in class 25  : 1350.0 	 Traffic Signal Label : Road work
    No. of Images in class 26  : 540.0 	 Traffic Signal Label : Traffic signals
    No. of Images in class 27  : 210.0 	 Traffic Signal Label : Pedestrians
    No. of Images in class 28  : 480.0 	 Traffic Signal Label : Children crossing
    No. of Images in class 29  : 240.0 	 Traffic Signal Label : Bicycles crossing
    No. of Images in class 30  : 390.0 	 Traffic Signal Label : Beware of ice/snow
    No. of Images in class 31  : 690.0 	 Traffic Signal Label : Wild animals crossing
    No. of Images in class 32  : 210.0 	 Traffic Signal Label : End of all speed and passing limits
    No. of Images in class 33  : 599.0 	 Traffic Signal Label : Turn right ahead
    No. of Images in class 34  : 360.0 	 Traffic Signal Label : Turn left ahead
    No. of Images in class 35  : 1080.0 	 Traffic Signal Label : Ahead only
    No. of Images in class 36  : 330.0 	 Traffic Signal Label : Go straight or right
    No. of Images in class 37  : 180.0 	 Traffic Signal Label : Go straight or left
    No. of Images in class 38  : 1860.0 	 Traffic Signal Label : Keep right
    No. of Images in class 39  : 270.0 	 Traffic Signal Label : Keep left
    No. of Images in class 40  : 300.0 	 Traffic Signal Label : Roundabout mandatory
    No. of Images in class 41  : 210.0 	 Traffic Signal Label : End of no passing
    No. of Images in class 42  : 210.0 	 Traffic Signal Label : End of no passing by vehicles over 3.5 metric tons


### C. Shuffle the dataset

### Model Architecture

mu = 0
sigma = 0.1
dropout = 0.75
EPOCHS = 50
BATCH_SIZE = 256

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

- [x] Train your model here.
- [x] Calculate and report the accuracy on the training and validation set.
- [x] Tuning the final model architecture 
- [x] the accuracy on the test set should be calculated and reported.

    Training...
    
    EPOCH 1 ...
    Validation Accuracy = 0.684
    Train Accuracy = 0.743
    
    EPOCH 2 ...
    Validation Accuracy = 0.819
    Train Accuracy = 0.886
    ...
    ...
    ...
        
    EPOCH 48 ...
    Validation Accuracy = 0.962
    Train Accuracy = 0.996
    
    EPOCH 49 ...
    Validation Accuracy = 0.959
    Train Accuracy = 0.996
    
    EPOCH 50 ...
    Validation Accuracy = 0.959
    Train Accuracy = 0.997
    
    Model saved
    Test Accuracy = 0.943

![png](output_46_0.png)

---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, 
- [X] download at least five pictures of German traffic signs from the web
- [X] use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### A. Load and Output the Images

['14-stop.png', '33-turn_right_ahead.png', '15-no-vechicles.png', '12-priority-road.png', '.ipynb_checkpoints', '22-bumpy-road.png']

![png](output_50_2.png)
![png](output_51_0.png)

### B. Predict the Sign Type for Each Image

- [X] Run the predictions here and use the model to output the prediction and accuracy for each image.
- [X] Make sure to pre-process the images with the same pre-processing pipeline used earlier.

For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.


INFO:tensorflow:Restoring parameters from ./lenet
[14 33 15 12 22]
![png](output_54_1.png)


### C. Analyze Performance on images from external source
    Test Accuracy = 100.000 %


### D. Output Top 5 Softmax Probabilities For Each Image Found on the Web

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
# Calculate the top 5 softmax probabilities for each test images

Softmax Probability Distribution
![png](output_61_1.png)

![png](output_61_3.png)

![png](output_61_5.png)

![png](output_61_7.png)

![png](output_61_9.png)

# Printing the softmax probabilities
    ------------------------------------------------------------
    ------------------------------------------------------------
    
    	 True Label is: =    14:Stop                          
    
       14: Stop                           99.969%
       17: No entry                       0.031%
       32: End of all speed and passing limits 0.000%
       34: Turn left ahead                0.000%
       35: Ahead only                     0.000%
    ------------------------------------------------------------
    ------------------------------------------------------------
    
    	 True Label is: =    33:Turn right ahead              
    
       33: Turn right ahead               100.000%
       37: Go straight or left            0.000%
       14: Stop                           0.000%
        3: Speed limit (60km/h)           0.000%
       35: Ahead only                     0.000%
    ------------------------------------------------------------
    ------------------------------------------------------------
    
    	 True Label is: =    15:No vehicles                   
    
       15: No vehicles                    100.000%
       26: Traffic signals                0.000%
       36: Go straight or right           0.000%
       35: Ahead only                     0.000%
        9: No passing                     0.000%
    ------------------------------------------------------------
    ------------------------------------------------------------
    
    	 True Label is: =    12:Priority road                 
    
       12: Priority road                  100.000%
       30: Beware of ice/snow             0.000%
       32: End of all speed and passing limits 0.000%
       35: Ahead only                     0.000%
       40: Roundabout mandatory           0.000%
    ------------------------------------------------------------
    ------------------------------------------------------------
    
    	 True Label is: =    22:Bumpy road                    
    
       22: Bumpy road                     99.976%
       29: Bicycles crossing              0.022%
       26: Traffic signals                0.002%
       25: Road work                      0.000%
       31: Wild animals crossing          0.000%
    ------------------------------------------------------------

---

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry


    ---------------------------------------------------------------------------

    <ipython-input-46-dcb048557f2e> in <module>()
    ----> 1 outputFeatureMap(x[11], conv2, activation_min=-1, activation_max=-1 ,plt_num=1)

    RuntimeError: Attempted to use a closed Session.

    ---------------------------------------------------------------------------