import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

class Matcher:
    
    """
    Class for finding, matching and visualizing keypoints between two images.
    """
    
    def __init__(self, algo: str):
        
        self.algo = self._get_algo(algo)
        self.image1 = None
        self.image2 = None
        
        self.kp1 = None
        self.kp2 = None
        self.matches = None

    def _get_algo(self, algo: str):
        """
        Selects the feature detection and description algorithm.
        Supported algorithms: 'SIFT', 'ORB'.
        """
        
        if algo == "SIFT":
            
            return cv2.SIFT_create(
                nfeatures=500, 
                contrastThreshold=0.01, 
                edgeThreshold=15, 
                sigma=1.4
            )
        elif algo == "ORB":
            
            return cv2.ORB_create(
                nfeatures=1500, 
                scaleFactor=1.1, 
                nlevels=12, 
                edgeThreshold=15, 
                fastThreshold=30
            )
        else:
            raise ValueError(f"Algorithm {algo} is not supported. Use 'SIFT' or 'ORB'.")
        
    def _preprocess(self, image: np.ndarray) -> np.ndarray:

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        image = clahe.apply(image)
        image = cv2.GaussianBlur(image, (3, 3), 1.0) 
        
        return image

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Loads and preprocesses a single image.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        image = self._preprocess(image)
        
        return image

    def match(self, image1_path: str, image2_path: str) -> Tuple[list, list, list]:
        """
        Find matches between two images.
        """
        self.image1 = self.load_image(image1_path)
        self.image2 = self.load_image(image2_path)

        # keypoints and descriptors
        kp1, des1 = self.algo.detectAndCompute(self.image1, None)
        kp2, des2 = self.algo.detectAndCompute(self.image2, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) \
            if isinstance(self.algo, cv2.SIFT) else cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
        matches = bf.match(des1, des2)

        # sorting matches by distance
        self.matches = sorted(matches, key=lambda x: x.distance)
        self.kp1 = kp1
        self.kp2 = kp2
        
        return kp1, kp2, self.matches

    def plot_matches(self, n_matches: int = 30):
        """
        Plots the matches between two images.
        """
        if self.image1 is None or self.image2 is None or self.matches is None:
            raise ValueError("Images or matches not loaded. Use match() method before plotting.")

        # draw matches
        kp1, kp2, matches = self.kp1, self.kp2, self.matches
        match_img = cv2.drawMatches(self.image1, 
                                    kp1, 
                                    self.image2, 
                                    kp2, 
                                    matches[:n_matches], 
                                    None,  
                                    matchesThickness=2,
                                    matchColor=(0, 255, 0),
                                    singlePointColor=(255, 0, 0),
                                    flags=cv2.DrawMatchesFlags_DEFAULT)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(match_img)
        plt.axis("off")
        plt.show()