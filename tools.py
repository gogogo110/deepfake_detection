import os

ori_path = r'C:\Users\Administrator\Desktop\毕业设计\一些数据暂时放在C盘\task4\deepfake_and_faceshifter\test\0'
# tar_path = r'C:\Users\Administrator\Desktop\毕业设计\一些数据暂时放在C盘\二分类60\original_two_1\test\0'

ori_files = os.listdir(ori_path)
for file in ori_files:
    l = len(file)
    if l != 8:
        new_file = '0'*(8-l) + file
        # print(file)
        os.rename(ori_path+'\\'+file, ori_path + '\\'+new_file)
