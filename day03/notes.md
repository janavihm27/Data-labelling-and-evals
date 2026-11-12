1. What does an IoU of 1 mean? What does an IoU near 0.14 mean at a 0.5 threshold?

Ans: An IoU of 1 means both bounding boxes overlap perfectly and are identical
An IoU near 0.14 means the boxes overlap very little. Since 0.14 is below the threshold of 0.5 the boxes are classified as unmatched.

2. Why is an empty file different from a missing file?

Ans: An empty file means the image was checked and no object was found. It is a valid       negative training example.
A missing file means the annotation is unavailable or the image is not trainable.

3. What do the four numbers after the class ID mean?

The four numbers represent:
x_center: Normalized horizontal center of the bounding box.
y_center: Normalized vertical center of the bounding box.
width: Normalized width of the bounding box.
height: Normalized height of the bounding box.
They are normalized by dividing the pixel values by the image width or height making them values between 0 and 1.