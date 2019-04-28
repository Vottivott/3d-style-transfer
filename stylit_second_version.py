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

# Returns the number of performed assignments
def perform_good_assignments(offsets, reversed_offsets, a_size, b_size):
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
    sort_indices = np.argsort(errors)
    sorted_errors = errors[sort_indices]
    sorted_i_indices = i_indices[sort_indices]
    sorted_j_indices = j_indices[sort_indices]

    if len(errors) > 5:

        # Normalize
        yvals = np.array(sorted_errors) - sorted_errors[0]
        D_range = yvals[-1]
        yvals = yvals/D_range
        xvals = np.array(range(len(i_indices)))
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
        knee_point_index = int(knee_point_x * len(sorted_errors))
        plt.savefig("errors.png")
        plt.clf()
    else:
        knee_point_index = len(errors)

    # Perform the assignments that are before the knee point
    offsets[sorted_i_indices[:knee_point_index], sorted_j_indices[:knee_point_index], :] = temp_offsets[sorted_i_indices[:knee_point_index], sorted_j_indices[:knee_point_index], :]
    return knee_point_index

def get_offsets_all_pointing_at_same_point(a_size):
    offsets_y = np.zeros(a_size[:2]) - np.arange(a_size[0]).reshape((a_size[0], 1))
    offsets_x = np.zeros(a_size[:2]) - np.arange(a_size[1]).reshape((1, a_size[1]))
    best_d = np.zeros(a_size[:2])
    offsets_and_best_D = np.stack((offsets_y, offsets_x, best_d), 2)
    return offsets_and_best_D

if __name__ == "__main__":
    source, target, channel_weights = load_stylit_images()

    num_pyramid_levels = 7#6  # 6
    a_w, a_h = 400, 400
    a = target
    a_guides_original = np.copy(target[:,:,3:])
    b_original = source
    a = rescale_images(a, 0.5 ** (num_pyramid_levels - 1))


    for pyramid_level in range(num_pyramid_levels):
        b = rescale_images(b_original, 0.5 ** (num_pyramid_levels - 1 - pyramid_level))
        for iteration in range(6):
            patch_size = 5  # [23, 13, 5][iteration]
            if patch_size >= min(a_w, a_h, b.shape[0], b.shape[1]):
                continue
            patch_radius = int((patch_size - 1) / 2)
            a_size = np.array(a.shape[:2]) - (patch_size - 1)
            b_size = np.array(b.shape[:2]) - (patch_size - 1)
            offsets = get_offsets_all_pointing_at_same_point(a_size) #np.zeros((a_size[0], a_size[1], 3))
            num_target_patches = a_size[0] * a_size[1]
            num_assignments = 0
            while num_assignments/num_target_patches < 0.95:
                # Reversed NNF retrieval
                reversed_offsets = patchmatch(b, a, None, offsets[:,:,2] != 0, patchmatch_iterations=6, patch_size=patch_size,
                                     iteration_callback=None, offsets=None, channel_weights=channel_weights)
                # Sort potential patch assignments
                num_assignments += perform_good_assignments(offsets, reversed_offsets, a_size, b_size)
                print(str(int(100*num_assignments / num_target_patches)) + "% of patches assigned")
                show_image(reconstruct_image(offsets, a[:,:,:3], b[:,:,:3], patch_radius))
            if num_assignments != num_target_patches:
                # Solve remaining 5% with standard target-to-source patchmatch
                offsets = patchmatch(a, b, offsets[:,:,2] != 0, np.zeros(b_size), patchmatch_iterations=6, patch_size=patch_size,
                                     iteration_callback=None, offsets=offsets, channel_weights=channel_weights)
                print("Solved last assignments with standard patchmatch")
                show_image(reconstruct_image(offsets, a[:, :, :3], b[:, :, :3], patch_radius))
                plt.pause(1.0)
            a[:,:,:3] = reconstruct_image(offsets, a[:,:,:3], b[:,:,:3], patch_radius)

            # a[:,:,:3] = reconstruct_image(offsets, a[:,:,:3], b[:,:,:3], patch_radius)
            # print("Finished texture syntesis iteration " + str(iteration))
            # show_image(a[:,:,:3])
        save_image(a[:, :, :3], "output.png")
        rescaled_output = rescale_images(a[:,:,:3], 2.0)
        if pyramid_level < num_pyramid_levels-1:
            a = np.concatenate((rescaled_output, rescale_images(a_guides_original, rescaled_output.shape[:2])), 2)