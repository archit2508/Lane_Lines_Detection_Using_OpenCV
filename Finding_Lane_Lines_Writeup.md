# Finding Lane Lines on the Road

### Goals of the project - 
**1. To create a pipeline for detecting lane lines on a road**
**2. Apply that pipeline on a video stream to get continuous lane lines in the output video as we will desire in real world scenarios**


## PIPELINE

### 1. Loaded the test images
```python
import glob
images = glob.glob("test_images/*")

for test_image in images:
    file_name = test_image.split('/')[1]
    #PIPELINE
    cv2.imwrite("test_images_output/" + file_name,final_image)
```

### 2. Converted the image to grayscale
This was done to get monotonic coloured image so that strong change in intensity gradient can be detected easily to find edges later
```python
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
gray_image = grayscale(image)
```

### 3. Applied gaussian blur to get smooth edges later
This was done to smooth out the noise in the image to facilitate smooth edge detection
```python
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

blurred_image = gaussian_blur(gray_image, 5)
```

### 4. Applied canny to get edges in the image
This algo involves low and high thresholds of pixels which are used to detect edges in such a way that if an edge's pixel gradient value is higher than threshold, it is marked as strong edge, discarded if it is below threshold and weak edge if value lies between the 2 thresholds.
```python
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)
    
edges = canny(blurred_image, 50, 150)
```

### 5. Applied mask to get region of interest
I defined a set of vertices here to construct a polygon over the image to exlude all the pixels outside of polygon from being considered for lane detection.
This allows me to narrow down my search area for efficient results.
```python
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)                     #defining a blank mask to start with
    
    if len(img.shape) > 2:                        #defining a 3 channel or 1 channel color to fill the mask
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)     #returning the image only where mask pixels are nonzero
    return masked_image
    
imshape = image.shape
vertices = np.array([[(0,imshape[0]),(450, 320), (520, 320), (imshape[1],imshape[0])]], dtype=np.int32)
my_region = region_of_interest(edges, vertices)
```

### 6. Applied Hough transform
On the region of interest, I applied hough transform which on the basis of some parameters provides coordinates of potential line segments which can be constructed to later get the lane lines in the image after extrapolation
```python
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
    
rho = 2
theta = np.pi/180
threshold = 15
min_line_length = 40
max_line_gap = 40
line_image = hough_lines(my_region, rho, theta, threshold, min_line_length, max_line_gap)
```

### 7. Modifying Draw Lines function to extrapolate and get full lane lines 
1. Check slope for each set of coordinates
2. Store the coordinates in left and right array according to belonging of slope to left or right lane.
3. Pass all set of coordinates for left lane to polyplot function which will average out the positions of line segments given by these coordinates and return a single slope and intercept value for a single left lane line.
4. We perform similar set of steps for right lane.
5. After getting slope and intercept for both lines, we can now get coordinates for these 2 lines to draw full lane lines.
6. We already have y ccordinates for both lines as the top and bottom value of ROI.
7. Now we can use the slope and intercept deduced in previous steps to get x coordinates using polynomial expression.
8. Finally, we can use draw_line function to plot single left lane line and right lane line using the coordinates calculated in last step.
```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    x_left = []
    y_left = []
    x_right = []
    y_right = []
    y3 = imshape[0]
    y4 = imshape[0]*0.6
    y3 = int(y3)
    y4 = int(y4)
    xc = imshape[1]/2
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x1==x2:
                continue                              #To avoid infinit slope
                
            line_seg = np.polyfit((x1,x2),(y1,y2),1)

            if line_seg[0]<0 and x1<xc and x2<xc:     #negative slope implies left lane 
                x_left.append(x1)
                x_left.append(x2)
                y_left.append(y1)
                y_left.append(y2)
                
            if line_seg[0]>0 and x1>xc and x2>xc:     #positive slope implies right lane
                x_right.append(x1)
                x_right.append(x2)
                y_right.append(y1)
                y_right.append(y2)
        
    line_left = np.polyfit(x_left,y_left,1)           #Averaged the positions of all line segments in left lane
    line_right = np.polyfit(x_right,y_right,1)        #Averaged the positions of all line segments in right lane
             
    x1 = (y3-line_left[1])/line_left[0]                     #Calculating x coordinates for y coordinates lying at top and 
    x2 = (y4-line_left[1])/line_left[0]                     #bottom of our ROI for left lane to get a single full lane line
    cv2.line(img, (int(x1), y3), (int(x2), y4), color, 10)  #Drawing the final left lane line using above found coordinates
            
    x1 = (y3-line_right[1])/line_right[0]                   #Calculating x coordinates for y coordinates lying at top and 
    x2 = (y4-line_right[1])/line_right[0]                   #bottom of our ROI for right lane to get a single full lane line
    cv2.line(img, (int(x1), y3), (int(x2), y4), color, 10)  #Drawing the final right lane line using above found coordinates
```

### 8. Drawing the lines over original image
Finally I merged the lines drawn with original image using weighted_img function
```python
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)
    
final_image = weighted_img(line_image, image, α=0.8, β=1., γ=0.)
```


## Applying the above pipeline on videos 
```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML

"Using my pipeline here"
def process_image(image):
    gray_image = grayscale(image)
    plt.imshow(image)

    blurred_image = gaussian_blur(gray_image, 5)

    edges = canny(blurred_image, 50, 150)

    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 320), (520, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    my_region = region_of_interest(edges, vertices)

    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_length = 40
    max_line_gap = 40
    line_image = hough_lines(my_region, rho, theta, threshold, min_line_length, max_line_gap)

    final_image = weighted_img(line_image, image, α=0.8, β=1., γ=0.)
    result = final_image

    return result
    
white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)                  #This statement will use my pipeline for the series of images in the video
%time white_clip.write_videofile(white_output, audio=False)

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)

```


## Shortcomings in the project
1. Although the pipeline created in this project can detect straight lane lines on a road image but this pipeline won't give good results when the car encounters sharp turns. 
2. Moreover, it can be difficult for this pipeline to detect lane lines in case of heavy traffic which may block the lane lines present on the road from computer vision. Thus, this pipeline gives a necessity to have as minimum traffic as possible for clear vision of lines.

## Scope of Improvement
1. This pipeline can be improved by utilising the multi-polynomial functionality of polyplot function to detect and draw curved lane lines. 
2. Further improvement can be done by using previous frames in videos for smooth lane detection in case the current frame doesnt give clear vision of lines.