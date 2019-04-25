from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    #data = np.asarray(img, dtype="int32")
    data = np.asarray(img, dtype="float32") / 255
    return data

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
    print(img)
    plt.imshow(img)
    plt.show()



