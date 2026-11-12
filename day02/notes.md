1. What does FPS mean?

Ans: FPS stands for Frames Per Second. It tells us how many frames or pictures a video contains per second.
For example: a video with 30 FPS contains 30 frames every second. 

2. Why can timestamp × fps select the wrong picture?

Ans: An annotation timestamp and a video timestamp may use different clocks. If the annotation clock is behind the video clock, we need to add an offset before converting the timestamp into a frame index

The correct formula is:
    Frame Index = (Annotation Timestamp + Offset) × FPS
Without the offset, timestamp × FPS selects the wrong frame

3. What is the offset for clip_a, in seconds and in frames?

Ans: The offset for clip_a is +0.2 seconds.
At 30 FPS:
Offset in frames = 0.2 × 30 = 6 frames.
The annotation clock is 6 frames behind the video clock so we add 6 frames to the annotation frame index.

4. Why did channel 0 not mean red?

Ans: OpenCV stores image colors in BGR format instead of RGB.
The channel order is:
    Channel 0 = Blue
    Channel 1 = Green
    Channel 2 = Red