import dlib
import cv2
import numpy as np
import time
import os
from matplotlib import path


def extract_ROI_points(frame, detector, predictor):
    '''
    :param frame:
    :param detector:
    :param predictor:
    :return: bool, ROI_landmarks
    '''
    pointss = []
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人脸数rects,时间主要用在人脸检测上,自动使用概率高的人脸
    rects = detector(img_gray, 1)
    if len(rects) > 0:
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[0]).parts()])
        for idx, point in enumerate(landmarks):
            pointss.append((point[0, 0], point[0, 1]))

        # 附加8个点
        pointss.append((int((pointss[2][0] + pointss[41][0]) / 2), (int((pointss[2][1] + pointss[41][1]) / 2))))
        pointss.append((int((pointss[41][0] + pointss[31][0]) / 2), (int((pointss[41][1] + pointss[31][1]) / 2))))
        pointss.append((int((pointss[46][0] + pointss[35][0]) / 2), (int((pointss[46][1] + pointss[35][1]) / 2))))
        pointss.append((int((pointss[46][0] + pointss[14][0]) / 2), (int((pointss[46][1] + pointss[14][1]) / 2))))
        pointss.append(
            (int((pointss[3][0] * 2 + pointss[32][0]) / 3), (int((pointss[3][1] * 2 + pointss[32][1]) / 3))))
        pointss.append(
            (int((pointss[3][0] + pointss[32][0] * 2) / 3), (int((pointss[3][1] + pointss[32][1] * 2) / 3))))
        pointss.append(
            (int((pointss[35][0] * 2 + pointss[13][0]) / 3), (int((pointss[35][1] * 2 + pointss[13][1]) / 3))))
        pointss.append(
            (int((pointss[35][0] + pointss[13][0] * 2) / 3), (int((pointss[35][1] + pointss[13][1] * 2) / 3))))
        pointss.append((int((pointss[33][0] + pointss[50][0]) / 2), (int((pointss[33][1] + pointss[50][1]) / 2))))
        pointss.append((int((pointss[33][0] + pointss[52][0]) / 2), (int((pointss[33][1] + pointss[52][1]) / 2))))
        ret = True
    else:
        ret = False

    return ret, pointss


def extract_face_ROIs(points):
    # 手动排三角形ROI,12ROI
    # ROI之间相邻，使得之后使用相关性的时候更直观
    # res =  [[points[5], points[6], points[15]],
    #         [points[6], points[7], points[15]],
    #         [points[7], points[8], points[15]],
    #         [points[8], points[10], points[15]],
    #         [points[8], points[9], points[10]],
    #         [points[0], points[9], points[10]],
    #         [points[0], points[1], points[10]],
    #         [points[1], points[10], points[11]],
    #         [points[1], points[2], points[11]],
    #         [points[2], points[3], points[11]],
    #         [points[3], points[4], points[11]],
    #         [points[10], points[11], points[15]]]

    # 手动排列ROI，可为多边形,15ROI
    res = [[points[0], points[36], points[68], points[1]],
           [points[36], points[41], points[40], points[39], points[69], points[68]],
           [points[39], points[27], points[30], points[69]],
           [points[27], points[42], points[70], points[30]],
           [points[42], points[47], points[46], points[45], points[71], points[70]],
           [points[45], points[16], points[15], points[7]],
           [points[1], points[68], points[72], points[3], points[2]],
           [points[68], points[69], points[73], points[72]],
           [points[69], points[30], points[33], points[73]],
           [points[30], points[70], points[74], points[33]],
           [points[70], points[71], points[75], points[74]],
           [points[71], points[15], points[14], points[13], points[75]],
           [points[3], points[72], points[48], points[4]],
           [points[72], points[73], points[61], points[48]],
           [points[73], points[33], points[62], points[61]],
           [points[33], points[74], points[63], points[62]],
           [points[74], points[75], points[54], points[63]],
           [points[75], points[13], points[12], points[54]],
           [points[4], points[48], points[67], points[6], points[5]],
           [points[67], points[66], points[57], points[8], points[7], points[6]],
           [points[65], points[66], points[57], points[8], points[9], points[10]],
           [points[65], points[54], points[12], points[11], points[10]]
           ]

    return res


def min_x(pos):
    min_v = 10000
    for i, j in pos:
        if min_v > i:
            min_v = i
    return min_v


def min_y(pos):
    min_v = 10000
    for i, j in pos:
        if min_v > j:
            min_v = j
    return min_v


def max_x(pos):
    max_v = -1
    for i, j in pos:
        if max_v < i:
            max_v = i
    return max_v


def max_y(pos):
    max_v = -1
    for i, j in pos:
        if max_v < j:
            max_v = j
    return max_v


# 计算一个三角形区域中的像素均值,输入三点与图，输出三通道均值
def mean_pixelvalue_triangular(pos, frame):
    count = 0
    pix_val = np.array([0, 0, 0])
    p = path.Path([(pos[0][0], pos[0][1]), (pos[1][0], pos[1][1]), (pos[2][0], pos[2][1])])
    # 改进遍历的范围
    for i in range(min(pos[0][0], pos[1][0], pos[2][0]), max(pos[0][0], pos[1][0], pos[2][0]) + 1):
        for j in range(min(pos[0][1], pos[1][1], pos[2][1]), max(pos[0][1], pos[1][1], pos[2][1]) + 1):
            if p.contains_points([(i, j)]):
                pix_val += np.array(frame[j][i])  # 因为frame是480*854*3，而这里横轴是854，所以要倒着来。
                count += 1
    return pix_val / count


# 计算一个任意形状区域中的像素均值,输入点与图，输出三通道均值
def mean_pixelvalue_anyshape(pos, frame):
    count = 0
    pix_val = np.array([0, 0, 0])
    p = path.Path([(pos[i][0], pos[i][1]) for i in range(np.array(pos).shape[0])])
    # 改进遍历的范围
    for i in range(min_x(pos), max_x(pos) + 1):
        for j in range(min_y(pos), max_y(pos) + 1):
            if p.contains_points([(i, j)]):
                pix_val += np.array(frame[j][i])  # 因为frame是480*854*3，而这里横轴是854，所以要倒着来。
                count += 1
    if count == 0:
        return [0, 0, 0]
    return pix_val / count



# 提取一个视频ROI的平均颜色值,默认使用12ROI分割
def mean_color_extractor(name, display=False):
    '''
    :param name:
    :param dispaly:
    :return: mean_color
    '''

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    capture = cv2.VideoCapture(name)

    # 部分参数获取
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))

    num_ROI = 22
    mean_color = np.zeros((num_ROI, 3, frame_count))

    # 所有三通道都是bgr排列的
    for i in range(frame_count):
        # if i == 0:
        #     ret, frame = capture.read()  # 前者返回布尔值，后者返回帧矩阵,三通道array
        #     # 12个ROI区域,主要时间用在了extract_ROI_points中。
        #     #
        #     # if type == '05' or type == '06':
        #     #     frame = np.rot90(frame)
        #     # if type == '04':
        #     #     frame = np.rot90(frame)
        #     #     frame = np.rot90(frame)
        #     #     frame = np.rot90(frame)
        #
        #     img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     # rects = detector(img_gray, 1)
        #
        #     ret, points = extract_ROI_points(frame, detector, predictor)
        #     if ret:
        #         ROIs = extract_face_ROIs(points)
        #     else:
        #         break
        #
        #     for j in range(num_ROI):
        #         mean_color[j][0][i], mean_color[j][1][i], mean_color[j][2][i] = \
        #             mean_pixelvalue_anyshape(ROIs[j], frame)
        #
        #     print(i, mean_color[0][0][i])
        #     frame = frame.copy()
        #
        #     # 显示视频及ROI,观察其区域
        #     if display:
        #         for point in points:
        #             cv2.circle(frame, (point[0], point[1]), 1, (255, 255, 255), thickness=-1)
        #         cv2.imshow('frame', frame)
        #         if cv2.waitKey(1) == ord('q'):
        #             break
        # else:
        #     ret, frame = capture.read()  # 前者返回布尔值，后者返回帧矩阵,三通道array
        #     # 12个ROI区域,主要时间用在了extract_ROI_points中。
        #
        #     # if type == '05' or type == '06':
        #     #     frame = np.rot90(frame)
        #     # if type == '04':
        #     #     frame = np.rot90(frame)
        #     #     frame = np.rot90(frame)
        #     #     frame = np.rot90(frame)
        #
        #     # print(frame.shape)
        #     ret, points = extract_ROI_points(frame, detector, predictor)
        #     if ret:
        #         ROIs = extract_face_ROIs(points)
        #     else:
        #         break
        #
        #
        #     for j in range(num_ROI):
        #         mean_color[j][0][i], mean_color[j][1][i], mean_color[j][2][i] = \
        #             mean_pixelvalue_anyshape(ROIs[j], frame)
        #     print(i, mean_color[0][0][i])
        #     frame = frame.copy()
        #
        #     # 显示视频及ROI,观察其区域
        #     if display:
        #         for point in points:
        #             cv2.circle(frame, (point[0], point[1]), 1, (255, 255, 255), thickness=-1)
        #         cv2.imshow('frame', frame)
        #         if cv2.waitKey(1) == ord('q'):
        #             break
        ret, frame = capture.read()  # 前者返回布尔值，后者返回帧矩阵,三通道array

        # print(frame.shape)
        ret, points = extract_ROI_points(frame, detector, predictor)
        if ret:
            ROIs = extract_face_ROIs(points)
        else:
            break


        for j in range(num_ROI):
            mean_color[j][0][i], mean_color[j][1][i], mean_color[j][2][i] = \
                mean_pixelvalue_anyshape(ROIs[j], frame)
        print(i, mean_color[0][0][i])
        frame = frame.copy()

        # 显示视频及ROI,观察其区域
        if display:
            for point in points:
                cv2.circle(frame, (point[0], point[1]), 1, (255, 255, 255), thickness=-1)
            cv2.imshow('frame', frame)
            # key = cv2.waitKey(3000000)
            if cv2.waitKey(1) == ord('q'):
                break


    capture.release()
    cv2.destroyAllWindows()
    return mean_color


def main(database, target):
    video_files = os.listdir(database)[:]

    for idx, video_path in enumerate(video_files):
        print(video_path)
        # 输入视频路径，返回平均颜色array
        mean_color = mean_color_extractor(database + '\\' + video_path, True)

        # 去掉视频路径中的.avi, 将平均颜色序列存储为mean_color， 格式为num_ROI*3*num_frame
        np.save(target + '\\' + video_path[:-4], mean_color)



database = r'E:\FaceForensics++\original_sequences\youtube\c23\avi'
target = r'E:\FaceForensics++\original_sequences\youtube\c23\22ROI_mean_color'
main(database, target)
# database = r'E:\FaceForensics++\manipulated_sequences\Deepfakes\c23\avi'
# target = r'E:\FaceForensics++\manipulated_sequences\Deepfakes\c23\22ROI_mean_color'
# main(database, target)
# database = r'E:\FaceForensics++\manipulated_sequences\Face2Face\c23\avi'
# target = r'E:\FaceForensics++\manipulated_sequences\Face2Face\c23\22ROI_mean_color'
# main(database, target)
# database = r'E:\FaceForensics++\manipulated_sequences\FaceShifter\c23\avi'
# target = r'E:\FaceForensics++\manipulated_sequences\FaceShifter\c23\22ROI_mean_color'
# main(database, target)
# database = r'E:\FaceForensics++\manipulated_sequences\FaceSwap\c23\avi'
# target = r'E:\FaceForensics++\manipulated_sequences\FaceSwap\c23\22ROI_mean_color'
# main(database, target)
# database = r'E:\FaceForensics++\manipulated_sequences\NeuralTextures\c23\avi'
# target = r'E:\FaceForensics++\manipulated_sequences\NeuralTextures\c23\22ROI_mean_color'
# main(database, target)
# database = r'E:\HKBU-MARs_V1+\s3'
# target = r'E:\HKBU-MARs_V1+\session3_22ROI'
# main(database, target)



