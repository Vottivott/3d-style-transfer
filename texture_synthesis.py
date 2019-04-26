from patchmatch import *
from image_loading import *

def iteration_callback(iteration, scanline, offsets, a, b, patch_radius):
    if scanline is None:
        print("Patchmatch iteration " + str(iteration))
    else:
        print("   Scanline " + str(scanline))
    show_image(reconstruct_image(offsets, a, b, patch_radius))

num_pyramid_levels = 6#6
a_w, a_h = 400, 400
a = np.zeros((int(a_h * 0.5 ** (num_pyramid_levels-1)), int(a_w * 0.5 ** (num_pyramid_levels-1)), 3))
b_original = load_image("water.jpg")

for pyramid_level in range(num_pyramid_levels):
    b = rescale_images(b_original, 0.5 ** (num_pyramid_levels-1-pyramid_level))
    offsets = None
    for iteration in range(3):
        patch_size = [23, 13, 5][iteration]
        if patch_size >= min(a_w, a_h, b.shape[0], b.shape[1]):
            continue
        offsets = patchmatch(a, b, patchmatch_iterations = 4, patch_size=patch_size, iteration_callback=iteration_callback, offsets=offsets)
        a = reconstruct_image(offsets, a, b, int((patch_size-1)/2))
        print("Finished texture syntesis iteration " + str(iteration))
        show_image(a)
    a = rescale_images(a, 2.0)
