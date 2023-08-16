from PIL import Image
import numpy as np
import glob
import cv2
import os
import itertools
import random

def download_class_fundus(opt):
    opt.input_name = "fundus_train_numImages_" + str(opt.num_images) + "_" + str(opt.policy) \
    + "_indexdown" + str(opt.index_download) + ".jpg"
    print(opt.input_name)
    scale = opt.size_image
    num_images = opt.num_images

    def imsave(img, i):
        im = Image.fromarray(img.astype(np.uint8)).resize((scale, scale))
        im.save("Input/Images/fundus_train_numImages_" + str(opt.num_images) +"_" + str(opt.policy)
                + "_indexdown" + str(opt.index_download) + "_" + str(i) + ".jpg")

    if opt.mode == "train":
        images_path = glob.glob("train/*.jpg")
        for i, img_path in enumerate(images_path[:num_images]):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imsave(img, i)

        genertator0 = itertools.product((0,), (False, True), (-1, 1, 0), (-1,), (0,))
        genertator1 = itertools.product((0,), (False, True), (0, 1), (0, 1), (0, 1, 2, 3))
        genertator2 = itertools.product((1,), (False, True), (0,), (0,), (0, 1, 2, 3))
        genertator3 = itertools.product((0,), (False, True), (-1,), (1, 0), (0,))
        genertator4 = itertools.product((1,), (False,), (1, -1), (0,), (0,))
        genertator5 = itertools.product((1,), (False,), (0,), (1, -1), (0,))
        genertator = itertools.chain(genertator0, genertator1, genertator2, genertator3, genertator4, genertator5)
        lst = list(genertator)
        random.shuffle(lst)
        path_transform = "TrainedModels/" + str(opt.input_name)[:-4]
        if not os.path.exists(path_transform):
            os.mkdir(path_transform)
        np.save(path_transform + "/transformations.npy", lst)

    path = "fundus_test_scale" + str(scale)
    if not os.path.exists(path):
        os.mkdir(path)

    good_test_data = [resize_image(cv2.imread(img), scale) for img in glob.glob("test/good/*.jpg")]
    bad_test_data = [resize_image(cv2.imread(img), scale) for img in glob.glob("test/bad/*.jpg")]

    test_data = good_test_data + bad_test_data
    test_labels = [1] * len(good_test_data) + [0] * len(bad_test_data)

    np.save(path + "/fundus_data_test_" + str(scale) + "_" + str(opt.index_download) + ".npy", test_data)
    np.save(path + "/fundus_labels_test_" + str(scale) + "_" + str(opt.index_download) + ".npy", test_labels)

    opt.input_name = "fundus_train_numImages_" + str(opt.num_images) + "_" + str(opt.policy) + "_indexdown" + str(opt.index_download) + ".jpg"
    return opt.input_name

def resize_image(img, scale):
    im = Image.fromarray(img.astype(np.uint8)).resize((scale, scale))
    return np.array(im)
