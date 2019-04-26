from patchmatch import *
from image_loading import *

def iteration_callback(iteration, scanline, offsets, a, b, patch_radius):
    if scanline is None:
        print("Patchmatch iteration " + str(iteration))
    else:
        print("   Scanline " + str(scanline))
    show_image(reconstruct_image(offsets, a[:,:,:3], b[:,:,:3], patch_radius))

if __name__ == "__main__":
    source, target, channel_weights = load_stylit_images()

    num_pyramid_levels = 6  # 6
    a_w, a_h = 400, 400
    a = target
    a_guides_original = np.copy(target[:,:,3:])
    b_original = source
    a = rescale_images(a, 0.5 ** (num_pyramid_levels - 1))
    #a = np.zeros((int(a_h * 0.5 ** (num_pyramid_levels - 1)), int(a_w * 0.5 ** (num_pyramid_levels - 1)), 3))
    #b_original = load_image("gears.jpg")

    for pyramid_level in range(num_pyramid_levels):
        b = rescale_images(b_original, 0.5 ** (num_pyramid_levels - 1 - pyramid_level))
        offsets = None
        for iteration in range(3):
            patch_size = 5  # [23, 13, 5][iteration]
            if patch_size >= min(a_w, a_h, b.shape[0], b.shape[1]):
                continue
            offsets = patchmatch(a, b, patchmatch_iterations=4, patch_size=patch_size,
                                 iteration_callback=iteration_callback, offsets=offsets, channel_weights=channel_weights)
            a[:,:,:3] = reconstruct_image(offsets, a[:,:,:3], b[:,:,:3], int((patch_size - 1) / 2))
            print("Finished texture syntesis iteration " + str(iteration))
            show_image(a[:,:,:3])
        rescaled_output = rescale_images(a[:,:,:3], 2.0)
        if pyramid_level < num_pyramid_levels-1:
            a = np.concatenate((rescaled_output, rescale_images(a_guides_original, rescaled_output.shape[:2])), 2)