import os
import numpy as np
from PIL import Image
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy import interpolate
import random

# band-pass filter
fs = 30
bpf_div = 60 * fs / 2
b_BPF40220, a_BPF40220 = signal.butter(4, ([40 / bpf_div, 220 / bpf_div]), 'bandpass')


def bandpass_filter(sig):
    return signal.filtfilt(b_BPF40220, a_BPF40220, sig)


# Chrome-PPG extraction
skin_vec = [0.3841, 0.5121, 0.7682]
B, G, R = 0, 1, 2


def Chrome_PPG(mean_color, window, start_frame):
    col_c = np.zeros((3, window))
    for col in [B, G, R]:
        col_stride = mean_color[col, start_frame:start_frame + window]  # 选取window长度的片段
        # col_c[col] = (col_stride / (np.mean(col_stride)+1e-2))  # 去除信号中的线性趋势
        col_c[col] = col_stride - np.mean(col_stride)
    x_s = 3 * col_c[R] - 2 * col_c[G]
    y_s = 1.5 * col_c[R] + col_c[G] - 1.5 * col_c[B]
    xf = bandpass_filter(x_s)
    yf = bandpass_filter(y_s)
    nx = np.std(xf)
    ny = np.std(yf)
    alpha_chrom = nx / (ny+1e-2)

    S = xf - alpha_chrom * yf
    return S


# normalization, (x-mean)/std
def normalization(data):
    _range = np.max(data) - np.min(data)
    res = (data - np.min(data)) / (_range + 1e-6)
    return res


def add_0(x):
    if x <= 9:
        return '0' + str(x)
    else:
        return str(x)


def compute_rppg(mean_color):
    '''
    :param mean_color:
    :return: num_ROI * rPPG_signal
    '''
    shape = mean_color.shape
    num_roi = shape[0]
    frame = shape[2]

    res = np.zeros((num_roi, frame))
    for roi in range(num_roi):
        color_channel = mean_color[roi]
        ppg = Chrome_PPG(color_channel, frame, 0)
        res[roi] = ppg
    return res


def interp2d(map, target_len=300):
    '''
    use this function when total frames of the video is not 300.
    '''
    # print(map.shape)
    num_roi = len(map)
    source_len = map.shape[2]
    source_x = np.array([i for i in range(source_len)])
    target_x = np.array([i * (source_len - 1) / (target_len - 1) for i in range(target_len)])
    res = np.zeros((num_roi, map.shape[1], target_len))

    for roi in range(num_roi):
        f = interpolate.interp1d(source_x, map[roi,0], kind='cubic')
        res[roi,0,:] = f(target_x)
        f = interpolate.interp1d(source_x, map[roi, 1], kind='cubic')
        res[roi, 1, :] = f(target_x)
        f = interpolate.interp1d(source_x, map[roi, 2], kind='cubic')
        res[roi, 2, :] = f(target_x)
    # print(res.shape)
    return res


def generate_freq(map):
    num, l = map.shape
    res = np.zeros((num, l))
    for i in range(num):
        res[i] = np.abs(np.fft.fft(map[i]))
    return res




database1 = r'E:\FaceForensics++\original_sequences\youtube\c23\60ROI_mean_color'
database2 = r'E:\FaceForensics++\manipulated_sequences\Deepfakes\c23\60ROI_mean_color'
database3 = r'E:\FaceForensics++\manipulated_sequences\Face2Face\c23\60ROI_mean_color'
database4 = r'E:\FaceForensics++\manipulated_sequences\FaceShifter\c23\60ROI_mean_color'
database5 = r'E:\FaceForensics++\manipulated_sequences\FaceSwap\c23\60ROI_mean_color'
database6 = r'E:\FaceForensics++\manipulated_sequences\NeuralTextures\c23\60ROI_mean_color'

npy_list1 = os.listdir(database1)
npy_list2 = os.listdir(database2)
npy_list3 = os.listdir(database3)
npy_list4 = os.listdir(database4)
npy_list5 = os.listdir(database5)
npy_list6 = os.listdir(database6)

database_train = r'C:\Users\Administrator\Desktop\毕业设计\一些数据暂时放在C盘\task4\deepfake_and_faceshifter\train'
database_test = r'C:\Users\Administrator\Desktop\毕业设计\一些数据暂时放在C盘\task4\deepfake_and_faceshifter\test'

count = 0
for i in range(698):
    # 类别0
    filename1 = database1 + '\\' + npy_list1[i]
    npy = np.load(filename1)
    t = (npy.shape[2]-64)//64
    # t = npy.shape[2]//32
    for sec in range(t):
        tmp = npy[:, :, sec * 64:sec*64+128]
        rppg_tmp = normalization(np.nan_to_num(compute_rppg(tmp)))
        # print(rppg_tmp.shape)

        outputImg = Image.fromarray(rppg_tmp * 255.0).convert('L')
        outputImg.save(database_train + '\\0\\' + str(count) + '.jpg')
        count += 1

    # 类别1
    if i <= 450:
        filename2 = database2 + '\\' + npy_list2[i]
        npy = np.load(filename2)
        t = (npy.shape[2] - 64) // 64
        # t = npy.shape[2] // 32
        for sec in range(t):
            tmp = npy[:, :, sec * 64:sec * 64 + 128]
            rppg_tmp = normalization(np.nan_to_num(compute_rppg(tmp)))

            outputImg = Image.fromarray(rppg_tmp * 255.0).convert('L')
            outputImg.save(database_train + '\\1\\' + str(count) + '.jpg')
            count += 1
    else:
        filename4 = database4 + '\\' + npy_list4[i]
        npy = np.load(filename4)
        t = (npy.shape[2] - 64) // 64
        # t = npy.shape[2] // 32
        for sec in range(t):
            tmp = npy[:, :, sec * 64:sec * 64 + 128]
            rppg_tmp = normalization(np.nan_to_num(compute_rppg(tmp)))

            outputImg = Image.fromarray(rppg_tmp * 255.0).convert('L')
            outputImg.save(database_train + '\\1\\' + str(count) + '.jpg')
            count += 1

    # # 类别2
    # filename3 = database3 + '\\' + npy_list3[i]
    # npy = np.load(filename3)
    # t = npy.shape[2]
    # secs = t // 64
    # for sec in range(secs):
    #     tmp = npy[:, :, sec * 64:(sec + 1) * 64]
    #     rppg_tmp = normalization(np.nan_to_num(compute_rppg(tmp)))
    #
    #     outputImg = Image.fromarray(rppg_tmp * 255.0).convert('L')
    #     outputImg.save(database_train + '\\2\\' + str(count) + '.jpg')
    #     count += 1
    #
    # # 类别3
    # filename4 = database4 + '\\' + npy_list4[i]
    # npy = np.load(filename4)
    # t = npy.shape[2]
    # secs = t // 64
    # for sec in range(secs):
    #     tmp = npy[:, :, sec * 64:(sec + 1) * 64]
    #     rppg_tmp = normalization(np.nan_to_num(compute_rppg(tmp)))
    #
    #     outputImg = Image.fromarray(rppg_tmp * 255.0).convert('L')
    #     outputImg.save(database_train + '\\3\\' + str(count) + '.jpg')
    #     count += 1
    #
    # # 类别4
    # filename5 = database5 + '\\' + npy_list5[i]
    # npy = np.load(filename5)
    # t = npy.shape[2]
    # secs = t // 64
    # for sec in range(secs):
    #     tmp = npy[:,:,sec*64:(sec+1)*64]
    #     rppg_tmp = normalization(np.nan_to_num(compute_rppg(tmp)))
    #
    #     outputImg = Image.fromarray(rppg_tmp * 255.0).convert('L')
    #     outputImg.save(database_train + '\\4\\' + str(count) + '.jpg')
    #     count += 1
    #
    # # 类别5
    # filename6 = database6 + '\\' + npy_list6[i]
    # npy = np.load(filename6)
    # t = npy.shape[2]
    # secs = t // 64
    # for sec in range(secs):
    #     tmp = npy[:, :, sec * 64:(sec + 1) * 64]
    #     rppg_tmp = normalization(np.nan_to_num(compute_rppg(tmp)))
    #
    #     outputImg = Image.fromarray(rppg_tmp * 255.0).convert('L')
    #     outputImg.save(database_train + '\\5\\' + str(count) + '.jpg')
    #     count += 1

count = 0
for i in range(698,997):
    # 类别0
    filename1 = database1 + '\\' + npy_list1[i]
    npy = np.load(filename1)
    t = (npy.shape[2]-64)//64
    # print(filename1)
    # t = npy.shape[2] // 32
    for sec in range(t):
        tmp = npy[:, :, sec * 64:sec*64+128]
        rppg_tmp = normalization(np.nan_to_num(compute_rppg(tmp)))
        # rppg_tmp = interp2d(rppg_tmp)

        outputImg = Image.fromarray(rppg_tmp * 255.0).convert('L')
        outputImg.save(database_test + '\\0\\' + str(count) + '.jpg')
        count += 1

    # 类别1
    if i <=900:
        filename2 = database2 + '\\' + npy_list2[i]
        npy = np.load(filename2)
        t = (npy.shape[2] - 64) // 64
        # t = npy.shape[2] // 32
        for sec in range(t):
            tmp = npy[:, :, sec * 64:sec * 64 + 128]
            rppg_tmp = normalization(np.nan_to_num(compute_rppg(tmp)))

            outputImg = Image.fromarray(rppg_tmp * 255.0).convert('L')
            outputImg.save(database_test + '\\1\\' + str(count) + '.jpg')
            count += 1
    else:
        filename4 = database4 + '\\' + npy_list4[i]
        npy = np.load(filename4)
        t = (npy.shape[2] - 64) // 64
        # t = npy.shape[2] // 32
        for sec in range(t):
            tmp = npy[:, :, sec * 64:sec * 64 + 128]
            rppg_tmp = normalization(np.nan_to_num(compute_rppg(tmp)))

            outputImg = Image.fromarray(rppg_tmp * 255.0).convert('L')
            outputImg.save(database_test + '\\1\\' + str(count) + '.jpg')
            count += 1



    # # 类别2
    # filename3 = database3 + '\\' + npy_list3[i]
    # npy = np.load(filename3)
    # t = npy.shape[2]
    # secs = t // 64
    # for sec in range(secs):
    #     tmp = npy[:, :, sec * 64:(sec + 1) * 64]
    #     rppg_tmp = normalization(np.nan_to_num(compute_rppg(tmp)))
    #
    #     outputImg = Image.fromarray(rppg_tmp * 255.0).convert('L')
    #     outputImg.save(database_test + '\\2\\' + str(count) + '.jpg')
    #     count += 1
    #
    # # 类别3
    # filename4 = database4 + '\\' + npy_list4[i]
    # npy = np.load(filename4)
    # t = npy.shape[2]
    # secs = t // 64
    # for sec in range(secs):
    #     tmp = npy[:, :, sec * 64:(sec + 1) * 64]
    #     rppg_tmp = normalization(np.nan_to_num(compute_rppg(tmp)))
    #
    #     outputImg = Image.fromarray(rppg_tmp * 255.0).convert('L')
    #     outputImg.save(database_test + '\\3\\' + str(count) + '.jpg')
    #     count += 1
    #
    # # 类别4
    # filename5 = database5 + '\\' + npy_list5[i]
    # npy = np.load(filename5)
    # t = npy.shape[2]
    # secs = t // 64
    # for sec in range(secs):
    #     tmp = npy[:, :, sec * 64:(sec + 1) * 64]
    #     rppg_tmp = normalization(np.nan_to_num(compute_rppg(tmp)))
    #
    #     outputImg = Image.fromarray(rppg_tmp * 255.0).convert('L')
    #     outputImg.save(database_test + '\\4\\' + str(count) + '.jpg')
    #     count += 1
    #
    # # 类别5
    # filename6 = database6 + '\\' + npy_list6[i]
    # npy = np.load(filename6)
    # t = npy.shape[2]
    # secs = t // 64
    # for sec in range(secs):
    #     tmp = npy[:, :, sec * 64:(sec + 1) * 64]
    #     rppg_tmp = normalization(np.nan_to_num(compute_rppg(tmp)))
    #
    #     outputImg = Image.fromarray(rppg_tmp * 255.0).convert('L')
    #     outputImg.save(database_test + '\\5\\' + str(count) + '.jpg')
    #     count += 1
