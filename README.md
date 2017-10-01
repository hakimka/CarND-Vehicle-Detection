
# Writeup 

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to  HOG feature vector. 

* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.

* The training script is located [here](https://github.com/hakimka/CarND-Vehicle-Detection/blob/master/processImageFindCars.ipynb) https://github.com/hakimka/CarND-Vehicle-Detection/blob/master/processImageFindCars.ipynb

* The pipeline is run on a video stream  project_video.mp4  with  a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles

* Bounding box for vehicles detected as displayed on the submission [video](https://github.com/hakimka/CarND-Vehicle-Detection/blob/master/videoOut/project_video.mp4).


[//]: # (Image References)
[image1a]: ./writeup/car.png
[image1b]: ./writeup/notCar.png
[hsvCh0]: ./writeup/hsvCh0.jpg
[hsvCh1]: ./writeup/hsvCh1.jpg
[hsvCh2]: ./writeup/hsvCh2.jpg
[hog]:    ./writeup/hogSmall.jpg
[windowSlide]: ./writeup/windowSlide.jpg
[windowSlide2]: ./writeup/windowSlide2.jpg
[candidates1]: ./writeup/candidates1.jpg
[candidates2]: ./writeup/candidates2.jpg
[candidates3]: ./writeup/candidates3.jpg

[videoHeat]: ./writeup/videoFrameHeat.png

[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
  

---	

### Histogram of Oriented Gradients (HOG)

#### 1. HOG features from the training images.

The code for this step is contained in the third code cell of the IPython notebook .  

 	if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))      
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            #8) Append features to list
            #img_features.append(hog_features)    
            
        if spatial_feat == True and hist_feat == True and hog_feat == True:
            features.append(np.concatenate((spatial_features, hist_features,hog_features)))
        if spatial_feat == True and hist_feat == True and hog_feat == False:
            features.append(np.concatenate((spatial_features, hist_features)))
        if spatial_feat == True and hist_feat == False and hog_feat == True:
            features.append(np.concatenate((spatial_features, hog_features)))
        if spatial_feat == False and hist_feat == True and hog_feat == True:
            features.append(np.concatenate((hist_features, hog_features)))
        if spatial_feat == False and hist_feat == False and hog_feat == True:
            features.append( hog_features)

I started by reading in all the `vehicle` and `non-vehicle` images (cell 4).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1a]

![alt text][image1b]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space, channel 2. The channel displayed the cars as the most distinct segments of the image. 

![alt text][hsvCh1]

and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled on the following:

	hog_feat          = True # HOG features on or off
	hog_channel       = 1 # Can be 0, 1, 2, or "ALL"
	orient            = 9  # HOG orientations
	pix_per_cell      = 8 # HOG pixels per cell
	cell_per_block    = 4 # HOG cells per block

#### 3. Tained a classifier.

I trained a linear SVM using single_img_features() function. The function receives parameters to use spatial_features and histograms, as well the hog_features. Having these parameters helped to select needed features. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window created search inspection windows varying from size 64 to 154 pixels.
 	
	y_start_stop = [350, 800] # Min and max in y to search in slide_window
    
    winBase = 64
    winDelta  = 30
    windows = []
    for i in range(0,3):
        winH = winW = winBase + winDelta*i
        winGroup = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                        xy_window=(winH, winW), xy_overlap=(0.6, 0.5))
        windows.extend(winGroup)

The search windows are placed in the bottom half of the image. 
 


![alt text][windowSlide]

#### 2. Examples of test images to demonstrate how pipeline is working

The images are scanned with window blocks of 3 scales ranging from 64x64 upto 154X154. The color space for image representation was the second channel of HSV image. The searches of the SVM were performed on histogram feature and hog feature for each inspection window.  The following examples demonstrate the search results of the pipeline:

![alt text][candidates1] 
![alt text][candidates2]
![alt text][candidates3]

 
    
    
--- 

### Video Implementation

#### 1. Link to the final video output.  
Here's a [link to my video result](https://github.com/hakimka/CarND-Vehicle-Detection/blob/master/videoOut/project_video.mp4)


#### 2. Filtering for false positives and combining overlapping bounding boxes.

For each frame a list of candidate windows was generated as presented above. The pipeline locating cars in the frames as follows: 

    global heat    

    toReturn = np.copy(image)
    
    window_list = []

    window_list = getAllSidingWindows(image)
        
    filtered_list = list(filter(lambda x: accept_box(x, (0, image.shape[1], (image.shape[0]//2+50), image.shape[0]-100)), window_list))



    all_hot = search_windows(image, filtered_list, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    
    draw_image = np.copy(image)
    window_img = draw_boxes(draw_image, all_hot, color=(0, 0, 255), thick=6)      
    
    heat_img, labels, heat = createLabelsAndHeatMap(image, heat,all_hot, 2)  
    
    final_img       = draw_labeled_bboxes(np.copy(image), labels)       
    toReturn        = combineImages(final_img, window_img, heat_img)

Fist, I generate a list of random windows covering lower half of the image. The use "search_window" function to utlize features such as histogram and hog. At the time of calling the function, a lambda function "filter_list" clips off any potential candidate box that falls outside the image boundary. 

Then a heat map of potential candidates is generated. The heat carries suggestion areas where the findings might be. 
The 'createLabelsAndHeatMap" does the following:

	def createLabelsAndHeatMap(image, heat, all_hot, threshold=3):
    
    
	    # Add heat to each box in box list    
	    heat_cooled = cool_heat(heat)
	    heat = add_heat(heat_cooled,all_hot)  
	    
	    # # # # # # # # # # # # # # # # # # # # # # # # # 
	    # Apply threshold to help remove false positives    
	    #    H E A T M A P     T H R E S H O L D    
	    # # # # # # # # # # # # # # # # # # # # # # # # 
	    
	    heat = apply_threshold(heat,threshold)    
	    
	    # Visualize the heatmap when displaying    
	    heatmap = np.clip(heat, 0, 255)
	    
	    # Find final boxes from heatmap using label function
	    labels = label(heatmap)
	    return heatmap, labels, heat

"Cools" off the previous heat map, i.e. reduces the intensity of the heat point by 10%. Then applies new suggested areas. The thresholds all pixels that did not get sufficient "amount of heat." The logic behind this pipeline to make areas where the car was visible last several times. If the observed car rides with a relative same speed as the observer car, the observed car will reappear in the following frames.

The video demonstrates candidate hot windows as well as the heat map. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Video frame with candidates and the heatmap:

![alt text][videoHeat]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

# Issues

The main issue the pipeline does not work in a real time. To process each frame it took more than 2-3 seconds. 

Another problem the algorithm produces a lot of false positives. The left wall in the video was identified quite a bit as "car presence." 

# Possible Failures
I think the existing approach will fail on variety of ambient lighting conditions. As well as presence of trucks, motor



# A possible approach

One way to beef up the car detection would be to use DNN. Another way to speed up the algorithm would be to please search windows along the lanes and scale them accordingly long the distance from the observer car.   

![alt text][windowSlide2]

But this approach itself needs some more work with be able to place the search inspection windows along the road lanes. The search boxes need to be able to follow the lane curves.