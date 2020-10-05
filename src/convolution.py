import cv2
import numpy as np
from scipy import misc
from utils.ImageUtils import show_image, filter_image, max_pooling

#Original image of size (512,512)
i = misc.ascent()
show_image(i)
i_transformed = filter_image(i)
show_image(i_transformed)

#Image size will be reduced to 1/4th the original size. ie. (256,256)
max_pooling_image =max_pooling(i_transformed)
show_image(max_pooling_image)
