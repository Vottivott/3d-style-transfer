import numpy as np

def init_offsets_and_best_D(a_size, b_size, patch_size):
    r = patch_size//2
    offsets_y = r + np.random.randint(b_size[0]-2*r, size=a_size[:2]) - np.arange(a_size[0]).reshape((a_size[0],1))
    offsets_x = r + np.random.randint(b_size[1]-2*r, size=a_size[:2]) - np.arange(a_size[1]).reshape((1,a_size[1]))
    best_d = np.zeros(a_size[:2])
    offsets_and_best_D = np.stack((offsets_y, offsets_x, best_d), 2)
    return offsets_and_best_D

def get_patch(image, x, y, patch_radius): # x,y specifies the pixel position of a patch center
    return image[int(y) - patch_radius: int(y) + patch_radius + 1,
           int(x) - patch_radius: int(x) + patch_radius + 1]
    # return image[int(y)-patch_radius : int(y)+patch_radius+1, int(x)-patch_radius : int(x)+patch_radius+1]
    # return image[y-patch_radius : y+patch_radius+1, x-patch_radius : x+patch_radius+1]

def D(patch_a, patch_b, channel_weights):
    diff = patch_a - patch_b
    if channel_weights is None:
        return np.sum(diff * diff)
    else:
        return np.inner(np.sum(diff * diff, (0,1)), channel_weights)

def get_correctly_cropped_patches(a, b, a_x, a_y, offset_x, offset_y, patch_radius):
    if 0 <= a_y - patch_radius and a_y + patch_radius < a.shape[0] and 0 <= a_x - patch_radius and a_x + patch_radius < a.shape[1]:
        return get_patch(a, a_x, a_y, patch_radius), get_patch(b, a_x+offset_x, a_y+offset_y, patch_radius), patch_radius, patch_radius, patch_radius, patch_radius
    else:
        # Handle regions closer to the boundary than patch_radius
        up_patch_radius = patch_radius - max(patch_radius - a_y, 0)
        left_patch_radius = patch_radius - max(patch_radius - a_x, 0)
        down_patch_radius = patch_radius - max(patch_radius + a_y - (a.shape[0] - 1), 0)
        right_patch_radius = patch_radius - max(patch_radius + a_x - (a.shape[1] - 1), 0)
        a_patch, b_patch = [
            image[int(y - up_patch_radius): int(y + down_patch_radius) + 1,
            int(x - left_patch_radius): int(x + right_patch_radius) + 1] for (image, x, y) in
            ((a, a_x, a_y), (b, a_x + offset_x, a_y + offset_y))]
        return a_patch, b_patch, up_patch_radius, left_patch_radius, down_patch_radius, right_patch_radius

def calculate_D(a, b, a_x, a_y, offset_x, offset_y, patch_radius, channel_weights):
    a_patch, b_patch, _, _, _, _ = get_correctly_cropped_patches(a, b, a_x, a_y, offset_x, offset_y, patch_radius)
    return D(a_patch, b_patch, channel_weights)

def assign_initial_D(a, b, offsets, patch_radius, channel_weights):
    height, width = offsets.shape[:2]
    for i in range(height):
        y = i
        for j in range(width):
            x = j
            offsets[i,j,2] = calculate_D(a, b, x, y, offsets[i,j,1], offsets[i,j,0], patch_radius, channel_weights)

def patch_omega(omega, x, y, patch_radius):
    return np.sum(omega[int(y)-patch_radius:int(y)+patch_radius+1, int(x)-patch_radius:int(x)+patch_radius+1])

def add_to_omega(omega, x, y, patch_radius, addition):
    omega[int(y)-patch_radius:int(y)+patch_radius+1, int(x)-patch_radius:int(x)+patch_radius+1] += addition

def try_patch(offsets, a, b, x, y, try_offset_x, try_offset_y, patch_radius, channel_weights, omega, omega_best, uniformity_weight, patch_size):
    D_using_try_offset = calculate_D(a, b, x, y, try_offset_x, try_offset_y, patch_radius, channel_weights)  # (Can optimize this)

    curr_occ = patch_omega(omega, x + offsets[y,x,1], y + offsets[y,x,0], patch_radius)/(patch_size*patch_size)/omega_best
    new_occ = patch_omega(omega, x + try_offset_x, y + try_offset_y, patch_radius)/(patch_size*patch_size)/omega_best
    if D_using_try_offset + uniformity_weight*new_occ < offsets[y, x, 2] + uniformity_weight*curr_occ:
        add_to_omega(omega, x + offsets[y,x,1], y + offsets[y,x,0], patch_radius, -1)
        add_to_omega(omega, x + try_offset_x, y + try_offset_y, patch_radius, +1)
        offsets[y, x, :] = try_offset_y, try_offset_x, D_using_try_offset


def propagate(offsets, i, j, delta_i, delta_j, a, b, x, y, patch_radius, channel_weights, omega, omega_best, uniformity_weight, patch_size):
    neighbor_offset_y, neighbor_offset_x, neighbor_best_D = offsets[i + delta_i, j + delta_j]
    if y+neighbor_offset_y-patch_radius < 0 or y+neighbor_offset_y+patch_radius >= b.shape[0] \
        or x+neighbor_offset_x-patch_radius < 0 or x+neighbor_offset_x+patch_radius >= b.shape[1]:
        return # Patch ends up outside B when using the neighbor's offset, so don't use it

    try_patch(offsets, a, b, x, y, neighbor_offset_x, neighbor_offset_y, patch_radius, channel_weights, omega, omega_best,
              uniformity_weight, patch_size)

def random_search(offsets, i, j, search_windows, a, b, x, y, patch_radius, channel_weights, omega, omega_best, uniformity_weight, patch_size):
    min_offset = -np.array([i, j]) + patch_radius
    max_offset = min_offset + np.array(b.shape[:2]) - 2*patch_radius
    for window_radius in search_windows:
        min_window = np.maximum(min_offset, offsets[i, j, :2] - window_radius)
        max_window = np.minimum(max_offset, offsets[i, j, :2] + window_radius)
        try_offset_y = np.random.randint(min_window[0], max_window[0])
        try_offset_x = np.random.randint(min_window[1], max_window[1])

        try_patch(offsets, a, b, x, y, try_offset_x, try_offset_y, patch_radius, channel_weights, omega,
                  omega_best,
                  uniformity_weight, patch_size)

def init_random_offsets(a, b, patch_size, channel_weights):
    offsets = init_offsets_and_best_D(a.shape, b.shape, patch_size)
    assign_initial_D(a, b, offsets, patch_size//2, channel_weights)
    return offsets

def upscaled_offsets(offsets, a, b, patch_size, channel_weights):
    upsize = np.ones((2,2))
    new_offsets = 2 * np.stack((np.kron(offsets[:,:,0], upsize),
                                np.kron(offsets[:,:,1], upsize),
                                np.zeros(np.array(offsets.shape[:2])*2)), 2)
    assign_initial_D(a, b, new_offsets, patch_size//2, channel_weights)
    return new_offsets


def init_omega(a, b, offsets, patch_radius):
    omega = np.zeros((offsets.shape[0], offsets.shape[1]), dtype=int)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            y = i+offsets[i,j,0]
            x = j+offsets[i,j,1]
            omega[int(y)-patch_radius:int(y)+patch_radius+1, int(x)-patch_radius:int(x)+patch_radius+1] += 1
    return omega

def patchmatch(a, b, offsets, patchmatch_iterations = 6, patch_size = 5, iteration_callback=None, scanline_callback_every_nth=50, channel_weights=None):
    patch_radius = patch_size // 2
    height, width = offsets.shape[:2]
    max_search_radius = max(height, width)
    window_size_ratio = 0.5
    num_search_windows = 1 + int(-np.log(max_search_radius)/np.log(window_size_ratio)) # All possible windows larger than 1 pixel
    print("num_search_windows = " + str(num_search_windows))
    search_windows = [max_search_radius * window_size_ratio ** i for i in range(num_search_windows)]
    omega = init_omega(a, b, offsets, patch_radius)
    omega_best = patch_size*patch_size * a.shape[0]*a.shape[1] / (b.shape[0]*b.shape[1])
    uniformity_weight = 3500

    for patchmatch_iteration in range(int(patchmatch_iterations/2)):
        # Right-down iteration
        for i in range(height):
            y = i
            for j in range(width):
                x = j
                # Propagation
                if j > 0:
                    propagate(offsets, i, j, 0, -1, a, b, x, y, patch_radius, channel_weights, omega, omega_best, uniformity_weight, patch_size)
                if i > 0:
                    propagate(offsets, i, j, -1, 0, a, b, x, y, patch_radius, channel_weights, omega, omega_best, uniformity_weight, patch_size)
                # Random search
                random_search(offsets, i, j, search_windows, a, b, x, y, patch_radius, channel_weights, omega, omega_best, uniformity_weight, patch_size)
            if i % scanline_callback_every_nth == 0:
                if iteration_callback is not None:
                    iteration_callback(patchmatch_iteration, i, offsets, a, b, patch_radius)
        # Left-up iteration
        for i in range(height-1,-1,-1):
            y = i
            for j in range(width-1,-1,-1):
                x = j
                # Propagation
                if j < width-1:
                    propagate(offsets, i, j, 0, 1, a, b, x, y, patch_radius, channel_weights, omega, omega_best, uniformity_weight, patch_size)
                if i < height-1:
                    propagate(offsets, i, j, 1, 0, a, b, x, y, patch_radius, channel_weights, omega, omega_best, uniformity_weight, patch_size)
                # Random search
                random_search(offsets, i, j, search_windows, a, b, x, y, patch_radius, channel_weights, omega, omega_best, uniformity_weight, patch_size)
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
        y = i
        for j in range(width):
            x = j
            _, b_patch, up_patch_radius, left_patch_radius, down_patch_radius, right_patch_radius = get_correctly_cropped_patches(a, b, x, y, offsets[i, j, 1], offsets[i, j, 0], patch_radius)
            reconstruction[int(y - up_patch_radius): int(y + down_patch_radius) + 1, int(x - left_patch_radius): int(x + right_patch_radius) + 1] += b_patch
            count[int(y - up_patch_radius): int(y + down_patch_radius) + 1, int(x - left_patch_radius): int(x + right_patch_radius) + 1] += 1
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
    patch_size = 5
    offsets = init_random_offsets(a, b, patch_size, None)
    # reconstruction = reconstruct_image(offsets, a[:, :, :3], b[:, :, :3], patch_size // 2)  # Initial reconstruction from the randomly initialized nearest-neighbor field
    # print(a.shape)
    # print(reconstruction.shape)
    # a = np.copy(reconstruction)
    patchmatch(a, b, offsets, patchmatch_iterations=4, patch_size=patch_size,
               iteration_callback=iteration_callback, channel_weights=None)
    # patchmatch(a, b, iteration_callback=iteration_callback)



