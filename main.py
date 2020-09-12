import matplotlib.pyplot as plt
import numpy as np
from numpy import (
    argmax,
    arange
)
from scipy.stats import multivariate_normal

from mnist_parser import group_mnist_images, get_training_files, get_testing_files
from my_parameter_estimator import estimate_properties


def test(mn_img_grouped, i):
    sums = []
    for pixel_ind in range(784):
        sum_pixels = 0
        for img in mn_img_grouped[i]:
            sum_pixels += img[pixel_ind]
        sums.append(sum_pixels / len(mn_img_grouped[i]))
    return sums


def calculate_f_ter(pkx, max_j, ks, target_d, fj_arr):
    for j in range(2, max_j):
        for d in range(0, target_d + 1):
            if fj_arr[d][j] is not None:
                continue
            else:
                fsum = 0
                for k in ks:
                    fsum += pkx[j - 1][k] * fj_arr[d - k][j - 1] if d - k >= 0 else 0
                fj_arr[d][j] = fsum
    return fj_arr


def calculate_f(pkx, j, r, ks, fj_arr):
    if r < 0:
        return 0
    if fj_arr[r][j] is not None:
        return fj_arr[r][j]
    if j == 1:
        if r > ks.size - 1 or r < 0:
            return 0
        else:
            return pkx[0][r]
    else:
        f_sum = 0
        for k in ks:
            f_sum += pkx[j - 1][k] * calculate_f(pkx, j - 1, r - k, ks, fj_arr)
        fj_arr[r][j] = f_sum
        return f_sum  # np.sum(pkx[j-1] * calculate_f(pkx, j-1, d))


def show_number(number):
    X = number.reshape([28, 28])
    plt.imshow(X)
    plt.show()


if __name__ == "__main__":

    # parse MNIST files
    mn_images, mn_labels = get_training_files('./')
    mn_images_testing = np.array(get_testing_files('./')[0])
    mn_labels_testing = np.array(get_testing_files('./')[1])

    mn_images_grouped = group_mnist_images(mn_images, mn_labels)

    n_testing = 500
    indices = np.random.randint(0, np.size(mn_images_testing, axis=0), n_testing)
    mn_images_testing = np.take(indices=indices, a=mn_images_testing, axis=0)
    mn_labels_testing = np.take(indices=indices, a=mn_labels_testing, axis=0)

    # mapping all pixel values to [0,1]
    mn_images_testing = mn_images_testing / 255

    images_count = np.size(mn_images_grouped[0])

    for i in range(len(mn_images_grouped)):
        mn_images_grouped[i] = (np.array(mn_images_grouped[i])) / 255

    means, stdevs, covs = estimate_properties(mn_images_grouped)
    # making sure the covariance matrix is invertible
    covs = covs + 0.1 * np.identity(784)

    # generate noise
    noise_means = np.zeros(784)
    noise_covs = np.identity(784) * 0.1
    # apply noise
    mn_images_testing += multivariate_normal(mean=noise_means, cov=noise_covs).rvs(n_testing)

    gau_digits_probabilities = np.empty(10)
    # probabilities of each image conditioned on each training image set: p(X=x|K=k)
    gau_images_probabilities_cond = np.empty(shape=(10, np.size(axis=0, a=mn_images_testing)))
    for i in range(0, 10):
        # P(X=x|K=k)
        gau_images_probabilities_cond[i] = (
            multivariate_normal(mean=means[i] + noise_means, cov=covs[i] + noise_covs).pdf(x=mn_images_testing))
        # estimated probability of each number: p(K=k)
        gau_digits_probabilities[i] = np.size(axis=0, a=mn_images_grouped[i]) / images_count
    # joint probabilities: p(x, k)
    gau_joint_probabilities = gau_images_probabilities_cond.T * gau_digits_probabilities.T
    # total probs of x: p(x)
    gau_images_probabilities = np.sum((gau_images_probabilities_cond.T * gau_digits_probabilities.T), axis=-1)
    # conditional probabilities of each number conditioned on each image: list of p(k|x)
    gau_digits_probabilities_cond = gau_joint_probabilities * (
        np.array([1 / gau_images_probabilities] * gau_digits_probabilities[0].size).transpose())

    fj_arr = np.array([[None] * (n_testing + 1)] * (n_testing * 9 + 1))
    for q in range(0, n_testing * 9 + 1):
        fj_arr[q][1] = gau_digits_probabilities_cond[0][q] if q < arange(0, 10).size else 0
    s = 0
    q = 0
    # loss function = |d*-d|
    while q < n_testing * 9 + 1:
        s += calculate_f(gau_digits_probabilities_cond, n_testing, q, arange(0, 10), fj_arr)
        if s >= 0.5:
            break
        q += 1

    mexp = 0
    for d in arange(0, n_testing * 9 + 1):
        mexp += calculate_f(gau_digits_probabilities_cond, n_testing, d, arange(0, 10), fj_arr) * d

    guesses = argmax(gau_digits_probabilities_cond, axis=1)
    print("Sum of MAP number guesses: ")
    print(np.sum(guesses))
    print("Absolute LF:")
    print(q)
    print("LSE (expectation):")
    print(mexp)
    print("True sum:")
    print(np.sum(mn_labels_testing))

    # PREVIOUS TASK
    # n = 100
    # digits_probabilities = array([0.2, 0.2, 0.1, 0.05, 0.05, 0.01, 0.01, 0.06, 0.16, 0.16])
    # images, digits = generate_images(
    #     ideal_digits=IDEAL_DIGITS,
    #     digits_number=n,
    #     vertical_scale=1,
    #     horizontal_scale=1,
    #     noise_level=p,
    #     digits_probabilities=digits_probabilities
    # )
    # images.shape = (n, 1, 5, 3)
    # bigxors = images ^ IDEAL_DIGITS
    # inverse_bigxors = 1 - bigxors
    # digits_probabilities = array([digits_probabilities] * n)
    #
    # elements = log(p) * bigxors + log(1 - p) * inverse_bigxors  # log because values with p could be too small
    #
    # # penalties = elements.sum(axis=(-2, -1)) + log(digits_probabilities)
    # # guesses = argmax(penalties, axis=1)
    #
    # # list of conditional probabilities of each pixel conditioned on k=K
    # pixel_probabilities_cond = (p ** bigxors) * ((1 - p) ** inverse_bigxors)
    # # list of p(x|k)
    # images_probabilities_cond = pixel_probabilities_cond.prod(axis=(-2, -1))
    # # list of p(x,k)
    # joint_probabilities = images_probabilities_cond * digits_probabilities[0].transpose()
    # # list of p(x)
    # images_probabilities = np.sum((images_probabilities_cond * digits_probabilities[0].transpose()), axis=-1)
    # # list of p(k|x)
    # digits_probabilities_cond = joint_probabilities * (
    #     np.array([1 / images_probabilities] * digits_probabilities[0].size).transpose())
    # # guessing the numbers by MAP (loss function = 1 if x=k, 0 otherwise)
    # guesses = argmax(digits_probabilities_cond, axis=1)
    # # list where the calculated Fj values will be stored
    # fjs = np.array([[None] * (n + 1)] * (n * 9 + 1))
    # # calculate the expectation of the sum
    # mexp = 0
    # for d in range(0, n * 9 + 1):
    #     fjs[d][1] = digits_probabilities_cond[0][d] if d < arange(0, 10).size else 0
    # for d in arange(0, n * 9 + 1):
    #     mexp += calculate_f(digits_probabilities_cond, n, d, arange(0, 10), fjs) * d
    # # c = calculate_f_ter(digits_probabilities_cond, n, n, arange(0, 10), n * 9 + 1)
    #
    # print("estimated numbers:")
    # print(guesses)
    # print("true numbers:")
    # print(digits)
    # print("estimated sum (Expectation):")
    # print(mexp)
    # print("true sum:")
    # print(sum(digits))
    # print("errors percentage:")
    # print((sum((guesses != digits) * 1)) * 100 / n)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # i don't remember why this is here
    # image = images[0]
    # xors = image[None, ...] ^ IDEAL_DIGITS
    # inverse_xors = 1 - xors
    # elements = log(p) * xors + log(1 - p) * inverse_xors
    # penalties = elements.sum(axis=(-2, -1))
    # print(argmax(penalties))
