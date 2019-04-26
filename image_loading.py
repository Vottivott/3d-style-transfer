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

def load_stylit_images(guidance_influence = 2.0, style = "style", channels = ("fullgi", "dirdif", "indirb"), scale=1.0):
    prefix = "stylit/source_"
    suffix = ".png"
    source = np.concatenate([load_image(prefix + channel + suffix) for channel in (style,) + channels], 2)
    prefix = "stylit/target_"
    target = np.concatenate([load_image(prefix + channel + suffix) for channel in channels], 2)
    output = np.zeros((target.shape[0],target.shape[1],3), dtype="uint8")
    target = np.concatenate((output, target), 2)
    channel_weights = np.array([1.0]*3 + [guidance_influence/float(len(channels))]*3*len(channels))
    if scale != 1.0:
        source = rescale_images(source, scale)
        target = rescale_images(target, scale)
    return source, target, channel_weights

if __name__ == "__main__":
    # img = load_image("gears.jpg")
    # # img = rescale_image(img, 0.1)
    # # images = np.concatenate((img,img),2)
    # img = rescale_images(img, 0.1)
    # print(img)
    # plt.imshow(img[:,:,:3])
    # plt.show()
    source, target, channel_weights = load_stylit_images(scale=0.5)
    for images in [source, target]:
        for i in range(int(images.shape[2]/3)):
            show_image(images[:,:,i*3:(i+1)*3])
            plt.pause(0.5)




