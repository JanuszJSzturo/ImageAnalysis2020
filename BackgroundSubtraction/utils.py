import cv2
import numpy as np
import glob
from tqdm import tqdm


def comparator(result, groundTruth):
    """
    Compares the background/foreground of 2 grayscale images as states in
    http://jacarini.dinf.usherbrooke.ca/datasetOverview/
    but with some simplification: the shadows and unknown are considered as good classification in either case.

    :param result: model background subtraction output
    :param groundTruth: expected result
    :return: tp(true positive),
            fp(false positive),
            fn(false negative),
            tn(true negative)
    """
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    groundTruth = cv2.cvtColor(groundTruth, cv2.COLOR_BGR2GRAY)

    FOREGROUND = 255
    BACKGROUND = 0
    UNKNOWN = 170
    SHADOW = 50

    bg_result = result == BACKGROUND
    fg_result = result == FOREGROUND

    # We will consider that UNKNOWN and SHADOW can be both considered as background or foreground
    bg_groundTruth = (groundTruth == BACKGROUND) | (groundTruth == UNKNOWN) | (groundTruth == SHADOW)
    fg_groundTruth = (groundTruth == FOREGROUND) | (groundTruth == UNKNOWN) | (groundTruth == SHADOW)

    # Model thinks it is foreground and it really is
    tp = sum(sum(np.bitwise_and(fg_result, fg_groundTruth)))

    # Model thinks it is foreground but it is not
    fp = sum(sum(np.bitwise_and(fg_result, np.bitwise_not(fg_groundTruth))))

    # Model thinks it is background but it is not
    fn = sum(sum(np.bitwise_and(bg_result, np.bitwise_not(bg_groundTruth))))

    # Model thinks it is background and it really is
    tn = sum(sum(np.bitwise_and(bg_result, bg_groundTruth)))

    return tp, fp, fn, tn


def loadImages(loadPath):
    """
    Load all images in the specified file and returns an array with all of them.
    :param loadPath: path where the images are located. End of path must be /*.{image extension}, e.g. /*.jpg
    :return: array with all images in the specified path
    """
    img_array = []
    for filename in tqdm(glob.glob(loadPath),desc='Loading images'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    return img_array


def exponentialFilter(img_array, alpha, savePath, init_frames, bgFilter='MEAN', morph_opening=True, morph_kernel=(2, 2)):
    """
    Calculates the backgroudn subtraction using the exponential filter approach.
    :param img_array: list of frames to do the brackgroudn substraction
    :param alpha: learning rate [0,1]. Value = 0 means background is not update, value = 1 means the new frame is set as
    background
    :param savePath: path where to save the resulting frames
    :param init_frames: the number of frames to estimate the initial background
    :param bgFilter: MEAN or MEDIAN to estimate the initial background
    :param morph_opening: True to apply morphological opening to the resulting background subtraction
    :param morph_kernel: kernel to be applied with the morphologial openning
    :return: nothing
    """
    # We get the shape of the images
    size = img_array[0].shape[0:2]

    # Initialize kernel for morphological transformation (opening)
    if(morph_opening):
        kernel = np.ones(morph_kernel, np.uint8)

    # Initial background calculation
    background = np.zeros(shape=size)
    h, w = size
    temp = np.zeros(shape=(init_frames, h, w))
    for i in range(init_frames):
        frame = img_array[i]
        temp[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Save the first init_frames in order to have the same amount of output items as input
        filepath = savePath
        filename = 'out' + str(i).zfill(6) + '.png'
        cv2.imwrite(filepath + filename, temp[i])
    if (bgFilter == 'MEAN'):
        background = temp.mean(axis=0).astype(np.uint8)
    elif (bgFilter == 'MEDIAN'):
        background = np.median(temp,axis=0).astype(np.uint8)


    # Algortihm aplication
    for i in tqdm(range(init_frames, len(img_array)), desc='Algorithm application'):
        # Take the next frame/image
        frame = img_array[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Substract the background from the frame to get the foreground (out)
        out = np.abs(frame - background)
        ret, out = cv2.threshold(out, 100, 255, cv2.THRESH_BINARY)
        if(morph_opening):
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
        # Calculate the new background
        background = ((1 - alpha) * background + alpha * frame).astype(np.uint8)

        # Save the result to the specified path
        filepath = savePath
        filename = 'out' + str(i).zfill(6) + '.png'
        cv2.imwrite(filepath + filename, out)


def MOG(img_array, savePath):
    """
    Calculates the background accorting to MOG algorithm
    “An improved adaptive background mixture model for real-time tracking with shadow detection” by P. KadewTraKuPong and R. Bowden in 2001

    :param img_array: list of sorted images corresponding to the frames of video
    :param savePath: path where to save the result
    :return:
    """
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    for i in tqdm(range(len(img_array)),desc='Algorithm application2'):
        frame = img_array[i]
        fgmask = fgbg.apply(frame)
        # Save the result to the specified path
        filepath = savePath
        filename = 'out' + str(i).zfill(6) + '.png'
        cv2.imwrite(filepath + filename, fgmask)


def MOG2(img_array, savePath):
    """
    Calculates the background according to MOG2 algorithm
    It is based on two papers by Z.Zivkovic, “Improved adaptive Gausian mixture model for background subtraction” in 2004
    and “Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction” in 2006
    :param img_array: list of sorted images corresponding to the frames of video
    :param savePath: path where to save the result
    :return:
    """
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    for i in tqdm(range(len(img_array)), desc='Algorithm application'):
        frame = img_array[i]
        fgmask = fgbg.apply(frame)

        # Save the result to the specified path
        filepath = savePath
        filename = 'out' + str(i).zfill(6) + '.png'
        cv2.imwrite(filepath + filename, fgmask)


def im2vid(img_path, name):
    """
    Creates video from corresponding frames
    :param img_path: path where the images are located. End of path must be /*.{image extension}, e.g. /*.jpg
    :param name: name of the video
    :return:
    """
    img_array = []

    for filename in tqdm(glob.glob(img_path), desc='Loading images'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), 24, size)  # Save video as mp4 format
    # out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'DIVX'), 24, size)

    for i in tqdm(range(len(img_array)), desc='Converting to video'):
        out.write(img_array[i])
    out.release()


def showVideo(path):
    """
    Plays on floating window the corresponding video
    :param path: path where the video is located.
    :return: nothing
    """
    cap = cv2.VideoCapture(path)

    if (cap.isOpened()):
        # leemos el primer frame
        ret, frame = cap.read()

        # mientras se haya podido leer el siguiente frame
        while(cap.isOpened() and ret):
            cv2.imshow('frame', frame)
            # leer siguiente frame
            ret, frame = cap.read()
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
    cap.release()
    cv2.destroyAllWindows()
