import numpy as np

def optimal_threshold(img):
    hist, _ = np.histogram(img, bins=256)
    pixels = img.shape[0] * img.shape[1]
    sum_pixel = np.sum(np.arange(256) * hist)
    sum_back = 0
    w_back = 0
    w_fore = 0
    var_max = 0
    threshold = 0

    for i in range(256):
        w_back += hist[i]
        if w_back == 0:
            continue
        w_fore = pixels - w_back
        if w_fore == 0:
            break
        sum_back += i * hist[i]
        mean_back = sum_back / w_back
        mean_fore = (sum_pixel - sum_back) / w_fore
        var_between = w_back * w_fore * (mean_back - mean_fore) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = i

    return threshold

def otsu_threshold(img):
    # Compute histogram
    hist, bin_edges = np.histogram(img, bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Total number of pixels
    total_pixels = np.sum(hist)

    # Compute probabilities and mean gray level values for each intensity level
    probabilities = hist / total_pixels
    mean_gray_levels = bin_centers * probabilities

    # Compute between-class variance for each threshold
    between_variances = np.zeros_like(probabilities)
    for t in range(1, 256):
        # Class probabilities and mean gray level values for classes separated by threshold t
        w0 = np.sum(probabilities[:t])
        w1 = 1 - w0
        if w0 == 0 or w1 == 0:
            continue
        mean0 = np.sum(mean_gray_levels[:t]) / w0
        mean1 = np.sum(mean_gray_levels[t:]) / w1
        between_variances[t] = w0 * w1 * (mean0 - mean1)**2

    # Find threshold that maximizes between-class variance
    threshold = np.argmax(between_variances)

    return threshold

def spectral_threshold(img):
    mean = np.mean(img)
    return mean

def localthresholding(img, block_size=15):
    rows, cols = img.shape
    binary_image = np.zeros((rows, cols))
    #img = cv.medianBlur(img,5)
    # go over blocks and apply a threshold over each block
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = img[i:i+block_size, j:j+block_size]
            block_mean = np.mean(block)
            binary_block = (block >= block_mean)
            binary_image[i:i+block_size, j:j+block_size] = binary_block
    return binary_image

def apply_threshold(img,threshold):
    binary_image = (img >= threshold)
    return binary_image.astype(np.float64)
    