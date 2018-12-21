# **Finding Lane Lines on the Road** 
### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

pipeline
    1. Use color selection to keep only yellow and white pixels of the original image
    2. Applies the Grayscale transform to the output of step 1
    3. Use kernel_size = 5 and apply Gaussian smoothing to the output of step 2
    4. Applies Canny Edge Detection to the output of step 3
    5. Choose the region of interest from the output of step 4
    6  Applies Hough Line Transform to the output of step 5
    
improve draw_lines()

Problem 1:
    Hough Line Transform returns a set of lines without telling us which lines belong to the left lane and which lines belong to the right lane.
Solution:
    check the slope, if the line slope < 0 => belongs to the left side
                     if the line slope > 0 => belongs to the right side
problem 2:
    How to decrease the noise in the line segments?
Solution:
    check the slope again, if abs(slope) < 0.1 skip this line segment
    slope = (y2 - y1) / (x2 - x1)
    if x2 == x1 set slope = 999 to avoid exception
    
problem 3:
    How to draw just one line for the left side of the lane, and one for the right?
Solution:
    After we solve problem 1 and 2, we divide the line segments into two groups. We also divide the start and end points of these line segments into two groups. We apply 1-degree polyfit to the points of right group and points of left group to get two expressions. Pass in start X(image_length * 0.48), start Y(image_width) to the left side expression to get the leftmost and rightmost points coordinate in the left lane. Pass in start X(image_length * 0.53), start Y(image_width) to right side expression to get leftmost and rightmost point of right lane. At last, we just draw the two line segments using coordinates we just got.


### 2. Identify potential shortcomings with your current pipeline
1. Can't identify dash line and solid line
2. If there is a white or yellow car/object in the region of interest, might have some problem
3. Many parameters need to tune if the weather not good or at night.

### 3. Suggest possible improvements to your pipeline
1. Use a more flexible way to decide the region of interest
2. Tune the parameters according to the input video quality parameters