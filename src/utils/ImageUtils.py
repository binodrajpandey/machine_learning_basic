import matplotlib.pyplot as plt
import numpy as np

def show_image(img):
    plt.grid(False)
    plt.gray()
    plt.axis("off")
    plt.imshow(img)
    plt.show()
def filter_image(i):
    i_transformed = np.copy(i)
    size_x = i_transformed.shape[0]
    size_y = i_transformed.shape[1]
    filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    # filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    # If all the digits in the filter don't add up to 0 or 1, you
    # should probably do a weight to get it to do so
    # so, for example, if your weights are 1,1,1 1,2,1 1,1,1
    # They add up to 10, so you would set a weight of .1 if you want to normalize them
    weight = 1
    for x in range(1, size_x - 1):
        for y in range(1, size_y - 1):
            output_pixel = 0.0
            output_pixel = output_pixel + (i[x - 1, y - 1] * filter[0][0])
            output_pixel = output_pixel + (i[x, y - 1] * filter[0][1])
            output_pixel = output_pixel + (i[x + 1, y - 1] * filter[0][2])
            output_pixel = output_pixel + (i[x - 1, y] * filter[1][0])
            output_pixel = output_pixel + (i[x, y] * filter[1][1])
            output_pixel = output_pixel + (i[x + 1, y] * filter[1][2])
            output_pixel = output_pixel + (i[x - 1, y + 1] * filter[2][0])
            output_pixel = output_pixel + (i[x, y + 1] * filter[2][1])
            output_pixel = output_pixel + (i[x + 1, y + 1] * filter[2][2])
            output_pixel = output_pixel * weight
            if (output_pixel < 0):
                output_pixel = 0
            if (output_pixel > 255):
                output_pixel = 255
            i_transformed[x, y] = output_pixel
    return i_transformed

def max_pooling(i_transformed):
    size_x = i_transformed.shape[0]
    size_y = i_transformed.shape[1]
    new_x = int(size_x / 2)
    new_y = int((size_y / 2))
    new_image = np.zeros((new_x, new_y))
    for x in range(0, size_x,2):
        for y in range(0, size_y,2):
            pixels = []
            pixels.append(i_transformed[x,y])
            pixels.append(i_transformed[x,y+1])
            pixels.append(i_transformed[x+1,y])
            pixels.append(i_transformed[x+1,y+1])
            pixels.sort(reverse=True)
            max = pixels[0]
            new_image[int(x/2),int(y/2)]=max
    return new_image