# Algorithm used here: https://paulbourke.net/miscellaneous/focuspeaking/

import numpy as np
import skimage as ski

# Read image
img = ski.io.imread("P1050612_edited.jpg")

# Grayscale
# TODO(mabay): Ensure image is rgb
gray = ski.color.rgb2gray(img)

# Gaussian
gauss_blur = ski.filters.gaussian(gray, sigma=0.4)

# Edge Detection [(sobel or roberts); chosen: sobel]
edges = ski.filters.sobel(gray)
print("Edge min:", edges.min())
print("Edge max:", edges.max())

# Threshold
threshold = 0.1
mask = edges > (threshold * edges.max())

# Apply to original image
img[mask, 0] = 0
img[mask, 1] = 255
img[mask, 2] = 0

# Save image
ski.io.imsave("final.jpg", img)
print("Final saved")
