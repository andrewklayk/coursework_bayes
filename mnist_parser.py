from mnist import MNIST


def get_training_files(path):
    mndata = MNIST(path)
    mndata.gz = True
    return mndata.load_training()


def get_testing_files(path):
    mndata = MNIST(path)
    mndata.gz = True
    return mndata.load_testing()


def group_mnist_images(mn_images, mn_labels):
    mn_images_grouped = [[], [], [], [], [], [], [], [], [], []]
    for i in range(len(mn_images)):
        mn_images_grouped[mn_labels[i]].append(mn_images[i])
    return mn_images_grouped
