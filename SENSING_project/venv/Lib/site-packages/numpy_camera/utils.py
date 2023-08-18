
import numpy as np


rgbw = np.array([0.2989, 0.5870, 0.1140])
bgrw = np.array([0.1140, 0.5870, 0.2989])

# rgb2gray = lambda im: np.dot(im, rgb2g).astype(np.uint8)
# bgr2gray = lambda im: np.dot(im, bgr2g).astype(np.uint8)

rgb2gray = lambda im: np.average(rgb, weights=rgbw, axis=2).astype(np.uint8)
bgr2gray = lambda im: np.average(rgb, weights=bgrw, axis=2).astype(np.uint8)

def gray2rgb(g): a=g.copy().T; return np.array([a,a,a],dtype=np.uint8).T
gray2bgr = lambda g: gray2rgb(g)
