import numpy as np



def init_offsets(a_size, b_size, patch_size):
    a_size = np.array(a_size[:2])-(patch_size-1)
    b_size = np.array(b_size[:2])-(patch_size-1)
    offsets_y = np.random.randint(b_size[0], size=a_size[:2]) - np.arange(a_size[0]).reshape((a_size[0],1))
    offsets_x = np.random.randint(b_size[1], size=a_size[:2]) - np.arange(a_size[1]).reshape((1,a_size[1]))
    offsets = np.stack((offsets_y, offsets_x), 2)
    #print(offsets[:,:,0]) # y offsets
    #print(offsets[:,:,1]) # x offsets
    return offsets

def init_offsets_and_best_D(a_size, b_size, patch_size):
    a_size = np.array(a_size[:2])-(patch_size-1)
    b_size = np.array(b_size[:2])-(patch_size-1)
    offsets_y = np.random.randint(b_size[0], size=a_size[:2]) - np.arange(a_size[0]).reshape((a_size[0],1))
    offsets_x = np.random.randint(b_size[1], size=a_size[:2]) - np.arange(a_size[1]).reshape((1,a_size[1]))
    best_d = np.zeros(a_size[:2])
    offsets_and_best_D = np.stack((offsets_y, offsets_x, best_d), 2)
    return offsets_and_best_D

def get_patch(image, x, y, patch_radius): # x,y specifies the pixel position of a patch center
    return image[int(y)-patch_radius : int(y)+patch_radius+1, int(x)-patch_radius : int(x)+patch_radius+1]
    # return image[y-patch_radius : y+patch_radius+1, x-patch_radius : x+patch_radius+1]

def D(patch_a, patch_b, channel_weights):
    diff = patch_a - patch_b
    if channel_weights is None:
        return np.sum(diff * diff)
    else:
        return np.inner(np.sum(diff * diff, (0,1)), channel_weights)


def calculate_D(a, b, a_x, a_y, offset_x, offset_y, patch_radius, channel_weights):
    return D(get_patch(a, a_x, a_y, patch_radius), get_patch(b, a_x+offset_x, a_y+offset_y, patch_radius), channel_weights)

def assign_initial_D(a, b, offsets, patch_radius, channel_weights):
    height, width = offsets.shape[:2]
    for i in range(height):
        y = i + patch_radius
        for j in range(width):
            x = j + patch_radius
            offsets[i,j,2] = calculate_D(a, b, x, y, offsets[i,j,1], offsets[i,j,0], patch_radius, channel_weights)

def propagate(offsets, i, j, delta_i, delta_j, a, b, x, y, patch_radius, channel_weights):
    neighbor_offset_y, neighbor_offset_x, neighbor_best_D = offsets[i + delta_i, j + delta_j]
    try:
        D_using_neighbor_offset = calculate_D(a, b, x, y, neighbor_offset_x, neighbor_offset_y, patch_radius, channel_weights)  # (Can optimize this)
    except ValueError:
        return # Patch ends up outside B when using the neighbor's offset, so don't use it
    if D_using_neighbor_offset < offsets[i, j, 2]:
        offsets[i, j, :] = neighbor_offset_y, neighbor_offset_x, D_using_neighbor_offset

def random_search(offsets, i, j, b_offsets_size, search_windows, a, b, x, y, patch_radius, channel_weights):
    min_offset = -np.array([i, j])
    max_offset = min_offset + b_offsets_size
    for window_radius in search_windows:
        min_window = np.maximum(min_offset, offsets[i, j, :2] - window_radius)
        max_window = np.minimum(max_offset, offsets[i, j, :2] + window_radius)
        try_offset_y = np.random.randint(min_window[0], max_window[0])
        try_offset_x = np.random.randint(min_window[1], max_window[1])
        try_D = calculate_D(a, b, x, y, try_offset_x, try_offset_y, patch_radius, channel_weights)
        if try_D < offsets[i, j, 2]:
            offsets[i, j, :] = try_offset_y, try_offset_x, try_D

def patchmatch(a, b, patchmatch_iterations = 6, patch_size = 5, iteration_callback=None, scanline_callback_every_nth=50, offsets=None, channel_weights=None):
    patch_radius = int((patch_size - 1) / 2)
    if offsets is None:
        offsets = init_offsets_and_best_D(a.shape, b.shape, patch_size)
    b_offsets_size = np.array(b.shape[:2])-(patch_size-1)
    height, width = offsets.shape[:2]
    max_search_radius = max(height, width)
    window_size_ratio = 0.5
    num_search_windows = 1 + int(-np.log(max_search_radius)/np.log(window_size_ratio)) # All possible windows larger than 1 pixel
    print("num_search_windows = " + str(num_search_windows))
    search_windows = (max_search_radius * window_size_ratio ** i for i in range(num_search_windows))

    assign_initial_D(a, b, offsets, patch_radius, channel_weights)

    for patchmatch_iteration in range(int(patchmatch_iterations/2)):
        # Right-down iteration
        for i in range(height):
            y = i + patch_radius
            for j in range(width):
                x = j + patch_radius
                # Propagation
                if j > 0:
                    propagate(offsets, i, j, 0, -1, a, b, x, y, patch_radius, channel_weights)
                if i > 0:
                    propagate(offsets, i, j, -1, 0, a, b, x, y, patch_radius, channel_weights)
                # Random search
                random_search(offsets, i, j, b_offsets_size, search_windows, a, b, x, y, patch_radius, channel_weights)
            if i % scanline_callback_every_nth == 0:
                if iteration_callback is not None:
                    iteration_callback(patchmatch_iteration, i, offsets, a, b, patch_radius)
        # Left-up iteration
        for i in range(height-1,-1,-1):
            y = i + patch_radius
            for j in range(width-1,-1,-1):
                x = j + patch_radius
                # Propagation
                if j < width-1:
                    propagate(offsets, i, j, 0, 1, a, b, x, y, patch_radius, channel_weights)
                if i < height-1:
                    propagate(offsets, i, j, 1, 0, a, b, x, y, patch_radius, channel_weights)
                # Random search
                random_search(offsets, i, j, b_offsets_size, search_windows, a, b, x, y, patch_radius, channel_weights)
            if i % scanline_callback_every_nth == 0:
                if iteration_callback is not None:
                    iteration_callback(patchmatch_iteration, i, offsets, a, b, patch_radius)
        if iteration_callback is not None:
            iteration_callback(patchmatch_iteration, None, offsets, a, b, patch_radius)
    return offsets

# Reconstruct an image using the calculated offsets
def reconstruct_image(offsets, a, b, patch_radius):
    reconstruction = np.zeros(a.shape)
    count = np.zeros(a.shape)
    height, width = offsets.shape[:2]
    for i in range(height):
        y = i + patch_radius
        for j in range(width):
            x = j + patch_radius
            patch = get_patch(b, x + offsets[i, j, 1], y + offsets[i, j, 0], patch_radius)
            reconstruction[y - patch_radius : y + patch_radius + 1, x - patch_radius : x + patch_radius + 1] += patch
            count[y - patch_radius: y + patch_radius + 1, x - patch_radius: x + patch_radius + 1] += 1
    reconstruction /= count
    return reconstruction.astype("uint8")

def iteration_callback(iteration, scanline, offsets, a, b, patch_radius):
    if scanline is None:
        print("Patchmatch iteration " + str(iteration))
    else:
        print("   Scanline " + str(scanline))
    show_image(reconstruct_image(offsets, a, b, patch_radius))


if __name__ == "__main__":
    # a_size=(2,3)
    # b_size=(2,4)
    # init_offsets(a_size, b_size)
    from image_loading import *
    a = load_image("test_a.jpg")
    b = load_image("test_b.jpg")
    patchmatch(a, b, iteration_callback=iteration_callback)



