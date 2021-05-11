import os
import numpy as np
file_path = r'C:\Users\Administrator\Desktop\毕业设计\一些数据暂时放在C盘\task4\deepfake_and_faceshifter\test\1'
name_list= os.listdir(file_path)
print(name_list)
num_list = [int(n[:-4]) for n in name_list]
num_list.sort()
# print(num_list)


res = []
pre = num_list[0]
cnt = 1
for i in range(1, len(num_list)):
    cur = num_list[i]
    if pre + 1 == cur:
        cnt += 1
    else:
        res.append(cnt)
        cnt = 1
    pre = cur
res.append(cnt)
print(res)
print(sum(res))
print(len(num_list))
res=np.array(res)
np.save(file_path+'\\'+'secs', res)


