import numpy as np


def estimate_properties(mn_img_grouped):
    means = []
    stdevs = []
    covs = []
    for num in range(len(mn_img_grouped)):
        means.append(np.mean(a=mn_img_grouped[num], axis=0))
        covs.append(np.cov(m=mn_img_grouped[num], rowvar=False, bias=False))
        stdevs.append(find_stdev(mn_img_grouped[num], means[num]))
    return np.array(means), np.array(stdevs), np.array(covs)


def find_means(mn_img_grouped):
    medians = []
    for listOfImagesOfNumber in mn_img_grouped:
        sums_in_number_map = map(sum, zip(*listOfImagesOfNumber))
        sums_in_number = []
        for x in sums_in_number_map:
            sums_in_number.append(x)
        medians.append(np.array(sums_in_number) / len(listOfImagesOfNumber))
    return medians


def find_stdev(list_of_images, means):
    list_of_images = np.array(list_of_images) - np.array(means)
    result = np.mean(axis=0, a=list_of_images ** 2)
    return np.sqrt(result)

    # for image in list_of_images:
    #     result += image ** 2
    # return np.sqrt(result / (len(list_of_images)))


def find_covs_one_num(list_of_images, means_in_num):
    list_of_images = np.array(list_of_images) - np.array(means_in_num)
    result = np.empty(shape=(list_of_images.shape[1], list_of_images.shape[1]))

    for image in list_of_images:
        result += np.outer(image, image)

    # for image in list_of_images:
    #     covs_in_img = np.full(shape=[784, 784], fill_value=np.inf)
    #     for x in range(len(image)):
    #         for y in range(len(image)):
    #             cov_xy = (covs_in_img[y][x] if (x < 784 and y < 784 and covs_in_img[y][x] != np.inf)
    #                       else ((image[x] - means_in_num[x]) * (image[y] - means_in_num[y])) / len(image))
    #             covs_in_img[x][y] = cov_xy
    #     covs.append(covs_in_img)
    # sums_in_number_map = map(sum, zip(*covs))
    # sums_in_number = []
    # for x in sums_in_number_map:
    #     sums_in_number.append(x)

    return result / len(list_of_images)
