# Image Matching Project

This project implements a tool for detecting, matching, and visualizing keypoints between pairs of satelite Level-2A images from [Sentinel-2](https://browser.dataspace.copernicus.eu/) using SIFT or ORB algorithms. 

## Features

- **Keypoint Detection and Matching:** 
  - Supports SIFT (Scale-Invariant Feature Transform) and ORB (Oriented FAST and Rotated BRIEF) algorithms.
  - Performs preprocessing using CLAHE and Gaussian Blurring to enhance feature detection.
  - Uses BFMatcher for finding matches.

- **Visualization:**
  - Visualizes image pairs with matched keypoints.
  - Displays individual images side-by-side for comparison.

### Usage
```python
from algo import Matcher

matcher = Matcher(algo='SIFT') # or 'ORB'

matcher.match_images('path/to/image1.jpg', 'path/to/image2.jpg')

matcher.visualize_matches(n=30) # customize the number of matches to display
```

### Project Structure
- **data/** - Two satellite images in every folder
- **algo.py** - Contains the `Matcher` class
- **demo.ipynb** - Interactive demo notebook
- **inference.py** - Visualization of original images and their loading 