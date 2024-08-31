import os
import numpy as np
import cv2
# 图片文件夹路径
#-----------------#
# 带文件夹的
#-----------------#
# pic_paths= r"D:\dataset\exposure"
# f=open(r'C:\Users\l\PycharmProjects\yolo\path\exposure/RoadIR.txt', 'w')
# n = 0
# for root,dirs,files in os.walk(pic_paths):
#     # print(root, dirs, files)
#      # 图片绝对路径
#     for i in files:
#         out_path = os.path.join(root, i)
#         print(out_path,n)
#         f.write(out_path + '\n')
#         n = n+1
# f.close()
#------------------#
#    直接是图像
#------------------#
path = r"E:\Datasets\FedFusion\test\visible/"
rootpath = os.listdir(path)
f = open('./path/test_TNO21_vis.txt', 'w')
# rootpath.sort(key=lambda x: int(x[-12:-4]))
for filename in rootpath:
    out_path = os.path.join(path, filename)
    f.write(out_path + '\n')
    print(filename)
f.close()
# #----------------#
#多张图像提取
#---------------#
# for j in range(1,230):
#     path = r"D:\dataset\expos/"+str(j)+'/'
#     rootpath = os.listdir(path)
#     rootpath.sort(key=lambda x: int(x[:-4]))
#     # print(len(rootpath))
#     temp = 1
#     for filename in rootpath:
#         if  temp == int(len(rootpath))-1:
#             a = filename
#         temp +=1
#     out_path = os.path.join(path, a)
#     print(out_path)
#     img = cv2.imread(out_path)
#     cv2.imwrite('D:/dataset/expos/train/max_2/' + str(j)+'.png',img)



