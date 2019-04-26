from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray(img, dtype="uint8")
    #data = np.asarray(img, dtype="int32")
    #data = np.asarray(img, dtype="float32") / 255
    return data

def rescale_image(image, scale):
    return imresize(image, scale)

def rescale_images(images, scale):
    return np.stack([imresize(images[:,:,i], scale) for i in range(images.shape[2])], 2)

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )
    #outimg = Image.fromarray(ycc_uint8, "RGB")
    #outimg.save("ycc.tif")

def show_image( image ):
    plt.ion()
    plt.imshow(image)
    plt.show()
    plt.pause(0.05)

if __name__ == "__main__":
    img = load_image("gears.jpg")
    # img = rescale_image(img, 0.1)
    # images = np.concatenate((img,img),2)
    img = rescale_images(img, 0.1)
    print(img)
    plt.imshow(img[:,:,:3])
    plt.show()



