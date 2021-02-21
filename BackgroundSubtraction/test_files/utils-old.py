import cv2
import numpy as np
import glob


def comparator(result, groundTruth):
    """
    Compares the background/foreground of 2 grayscale images as states in
    http://jacarini.dinf.usherbrooke.ca/datasetOverview/
    but with some simplification: the shadows and unknown are considered as good classification in either case.

    :param result: model background subtraction output
    :param groundTruth: expected result
    :return: tp, fp, fn, tn
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

    tp = sum(sum(np.bitwise_and(fg_result, fg_groundTruth)))
    fp = sum(sum(np.bitwise_and(fg_result, np.bitwise_not(fg_groundTruth))))
    fn = sum(sum(np.bitwise_and(bg_result, np.bitwise_not(bg_groundTruth))))
    tn = sum(sum(np.bitwise_and(bg_result, bg_groundTruth)))

    return tp, fp, fn, tn


# def im2im(loadPath, savePath):
#     """
#     :param loadPath: 'DATA/baseline/baseline/office/input/*.jpg'
#     :param savePath: 'DATA/baseline/results/office/'
#     :return:
#     """
#     img_array = []
#     for filename in glob.glob(loadPath):
#         img = cv2.imread(filename)
#         height, width, layers = img.shape
#         size = (width, height)
#         img_array.append(img)
#
#     for i in range(len(img_array)):
#         filepath = 'DATA/baseline/results/office/'
#         filename = 'out' + str(i).zfill(6)+'.jpg'
#         cv2.imwrite(filepath + filename, img_array[i])

def loadImages(loadPath):
    """
    Load all images in the specified file and returns an array with all of them.
    :param loadPath:
    :return: array with all images in the specified path
    """
    img_array = []
    for filename in glob.glob(loadPath):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    return img_array


# def saveImages (savePath):
#
#     """
#     :param savePath:
#     :return:
#     """
#     filepath = 'DATA/baseline/results/office/'
#     filename = 'out' + str(i).zfill(6) + '.jpg'
#     cv2.imwrite(filepath + filename, img_array[i])


def exponentialFilter(img_array, alpha, savePath):
    # We get the shape of the images
    im_shape = img_array[0].shape

    # Initialize kernel for morphology transformation
    kernel = np.ones((2, 2), np.uint8)

    # Number of initial frames to obtain the starting background
    init_frames = 20

    # learning rate [0,1]
    alpha = 0.05
    # Initial background calculation
    background = np.zeros(shape=im_shape[0:2])
    for i in range(init_frames):
        frame = img_array[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = background + frame

    background = background / init_frames
    background = background.astype(np.uint8)

    # Algortihm aplication
    for i in range(init_frames + 1, len(img_array)):
        # Take the next frame/image
        frame = img_array[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Substract the background from the frame to get the foreground (out)
        out = np.abs(frame - background)
        ret, out = cv2.threshold(out, 100, 255, cv2.THRESH_BINARY)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        # Calculate the new background
        background = ((1 - alpha) * background + alpha * frame).astype(np.uint8)

        # Save the result to the specified path
        filepath = savePath
        filename = 'out' + str(i).zfill(6) + '.jpg'
        cv2.imwrite(filepath + filename, out)


def MOG(img_array, savePath):
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    for i in range(len(img_array)):
        frame = img_array[i]
        fgmask = fgbg.apply(frame)

        # Save the result to the specified path
        filepath = savePath
        filename = 'out' + str(i).zfill(6) + '.jpg'
        cv2.imwrite(filepath + filename, fgmask)


def MOG2(img_array, savePath):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    for i in range(len(img_array)):
        frame = img_array[i]
        fgmask = fgbg.apply(frame)

        # Save the result to the specified path
        filepath = savePath
        filename = 'out' + str(i).zfill(6) + '.jpg'
        cv2.imwrite(filepath + filename, fgmask)


def im2vid_2(img_array, filename):
    """

    :param img_array:
    :param savePath:
    :param filename:
    :return:
    """
    size = img_array[0].shape[0:2]
    # out = cv2.VideoWriter('result_' + filename, cv2.VideoWriter_fourcc(*'mp4v'), 24, size) # Save video as mp4 format
    out = cv2.VideoWriter('result_' + filename, cv2.VideoWriter_fourcc(*'DIVX'), 24, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def im2vid(img_path, name):
    img_array = []
    for filename in glob.glob(img_path):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), 24, size) # Save video as mp4 format
    #out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'DIVX'), 24, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def showVideo (path):
    cap = cv2.VideoCapture(path)

    while (1):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# loadPath = 'DATA/baseline/baseline/highway/input/*.jpg'
# gtHighwayPath = 'DATA/baseline/baseline/highway/groundtruth/*.png'
# savePath = 'DATA/baseline/results/highway_MOG/'
# images = loadImages(loadPath)
# gt_hihgway_frames = loadImages(gtHighwayPath)
# # exponentialFilter(images,0.05,savePath)
# MOG(images, savePath)
# result_MOG = loadImages('DATA/baseline/results/highway_MOG/*jpg')

# tp =  np.zeros((len(result_MOG)))
# fn =  np.zeros((len(result_MOG)))
# fp =  np.zeros((len(result_MOG)))
# tn =  np.zeros((len(result_MOG)))
#
# for i in range(len(result_MOG)):
#     tp[i],fn[i],fp[i],tn[i] = comparator(result_MOG[i],gt_hihgway_frames[i])
#     print(tp[i],fn[i],fp[i],tn[i])

im2vid('DATA/baseline/results/highway_MOG/*.jpg', 'resultTest.mp4')

# im_gt = cv2.imread('DATA/baseline/baseline/highway/groundtruth/gt000684.png')
# im_gt_gray = cv2.cvtColor(im_gt, cv2.COLOR_BGR2GRAY)
# im_in = cv2.imread('DATA/baseline/baseline/office/input/in000001')
#
# # test_result = np.zeros(shape = (20,20))
# # plt.imshow(test_result, cmap="gray")
# # plt.show()
# #
# # test_gt = np.zeros(shape = (20,20))
# # test_gt[0:10,5:10]  = 255
# # plt.imshow(test_gt, cmap="gray")
# # plt.show()
# #
# # print(comparator(test_result,test_gt))
# # test_gt[15:17,12:17]  = 50
# # plt.imshow(test_gt, cmap="gray")
# # plt.show()
#
# test_result = np.zeros(shape = (3,3))
# test_result[0,1:3]  = 255
# test_result[1,2]  = 255
# plt.imshow(test_result, cmap="gray")
# plt.show()
#
# test_gt = np.zeros(shape = (3,3))
# test_gt[0:2,1:3]  = 255
# test_gt[1,1]  = 170
# plt.imshow(test_gt, cmap="gray")
# plt.show()
#
# print(comparator(test_result,test_gt))
