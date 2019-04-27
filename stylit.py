from patchmatch_with_masking import *
from image_loading import *
from scipy.optimize import curve_fit

def iteration_callback(iteration, scanline, offsets, a, b, patch_radius):
    if scanline is None:
        print("Patchmatch iteration " + str(iteration))
    else:
        print("   Scanline " + str(scanline))
    show_image(reconstruct_image(offsets, a[:,:,:3], b[:,:,:3], patch_radius))

def hyperbolic_function(x, a, b):
    return 1.0/(a - b * x)

def get_sorted_assignments(reversed_offsets, a_size, b_size):
    temp_offsets = np.zeros((a_size[0], a_size[1], 3))
    for i in range(b_size[0]):
        for j in range(b_size[1]):

            if temp_offsets[int(i+reversed_offsets[i,j,0]),int(j+reversed_offsets[i,j,1]),2] != 0: # If collision, keep the best assignment
                if reversed_offsets[i,j,2] < temp_offsets[int(i+reversed_offsets[i,j,0]),int(j+reversed_offsets[i,j,1]),2]:
                    temp_offsets[i + int(reversed_offsets[i, j, 0]), int(j + reversed_offsets[i, j, 1]), :2] = -reversed_offsets[i,j,:2]
                    temp_offsets[i + int(reversed_offsets[i, j, 0]), int(j + reversed_offsets[i, j, 1]), 2] = reversed_offsets[i,j,2]
            else:
                temp_offsets[i + int(reversed_offsets[i, j, 0]), int(j + reversed_offsets[i, j, 1]), :2] = -reversed_offsets[i, j,:2]
                temp_offsets[i + int(reversed_offsets[i, j, 0]), int(j + reversed_offsets[i, j, 1]), 2] = reversed_offsets[i, j, 2]

    i_indices, j_indices = np.nonzero(temp_offsets[:,:,2])
    errors = temp_offsets[i_indices, j_indices, 2]
    sorted_errors = sorted(errors)

    # Normalize
    yvals = np.array(sorted_errors) - sorted_errors[0]
    D_range = yvals[-1]
    yvals = yvals/D_range
    xvals = np.array(range(len(sorted_errors)))
    xvals = xvals/xvals[-1]

    plt.clf()
    plt.plot(xvals, yvals)

    print(xvals)
    print(yvals)
    (a,b),_ = curve_fit(hyperbolic_function, xvals, yvals, p0=(20, 19))
    plt.plot(xvals, hyperbolic_function(xvals, a, b))
    knee_point_x = -(1.0 / b) ** 0.5 + a / b
    knee_point_y = hyperbolic_function(knee_point_x, a, b)
    plt.scatter(knee_point_x, knee_point_y)
    knee_point_D = sorted_errors[0] + knee_point_y*D_range
    print("Knee point D at " + str(knee_point_D) + ", including " + str(int(knee_point_x*100)) + "% of the assignments")

    plt.savefig("errors.png")


if __name__ == "__main__":
    source, target, channel_weights = load_stylit_images()

    num_pyramid_levels = 6  # 6
    a_w, a_h = 400, 400
    a = target
    a_guides_original = np.copy(target[:,:,3:])
    b_original = source
    a = rescale_images(a, 0.5 ** (num_pyramid_levels - 1))


    for pyramid_level in range(num_pyramid_levels):
        b = rescale_images(b_original, 0.5 ** (num_pyramid_levels - 1 - pyramid_level))
        reversed_offsets = None
        for iteration in range(3):
            patch_size = 5  # [23, 13, 5][iteration]
            if patch_size >= min(a_w, a_h, b.shape[0], b.shape[1]):
                continue
            patch_radius = int((patch_size - 1) / 2)
            a_size = np.array(a.shape[:2]) - (patch_size - 1)
            b_size = np.array(b.shape[:2]) - (patch_size - 1)
            offsets = np.zeros((a_size[0], a_size[1], 3))
            # Reversed NNF retrieval
            reversed_offsets = patchmatch(b, a, offsets[:,:,2] != 0, patchmatch_iterations=4, patch_size=patch_size,
                                 iteration_callback=iteration_callback, offsets=reversed_offsets, channel_weights=channel_weights)
            # Sort potential patch assignments
            get_sorted_assignments(reversed_offsets, a_size, b_size)
            show_image(reconstruct_image(reversed_offsets, b[:,:,:3], a[:,:,:3], patch_radius))

            # a[:,:,:3] = reconstruct_image(offsets, a[:,:,:3], b[:,:,:3], patch_radius)
            # print("Finished texture syntesis iteration " + str(iteration))
            # show_image(a[:,:,:3])
        rescaled_output = rescale_images(a[:,:,:3], 2.0)
        if pyramid_level < num_pyramid_levels-1:
            a = np.concatenate((rescaled_output, rescale_images(a_guides_original, rescaled_output.shape[:2])), 2)