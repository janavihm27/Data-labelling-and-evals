1. Why is a frame shuffle a leak for video?
Frames from the same video are visually similar. If frames from the same source appear in both training and validation the model may memorize patterns instead of learning to generalize to unseen videos.

2. What does seed 42 buy you, and what does it not guarantee about quality?
Seed 42 makes the split reproducible, so we get the same split every time. But it does not guarantee that the split is balanced, representative or high quality.

3. When is accuracy a misleading score?
Accuracy can be misleading when the dataset is imbalanced. A model can achieve high accuracy by predicting only the majority class while missing important positive cases.

4. Which score answers "did we mark the right player?"
Application outcomes answer whether we marked the right player, such as Correct, Missed, Wrong instance and False positive.