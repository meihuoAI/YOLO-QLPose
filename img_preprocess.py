import cv2
import numpy as np
import os

# 输入和输出文件夹路径
input_folder = '/data1/home_data/chendi/datasets/grinding/images/val'  # 输入文件夹路径
output_folder = '/data1/home_data/chendi/datasets/grinding/images/preproposs'  # 输出文件夹路径

# 如果输出文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 只处理图像文件（可以根据文件扩展名过滤）
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 读取图像
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 检查图像是否读取成功
        if image is None:
            print(f"无法读取图像文件: {image_path}")
            continue

        # 预处理：高斯滤波去噪
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # 使用Canny边缘检测
        edges = cv2.Canny(blurred, 100, 200)

        # 获取边缘的梯度方向和梯度大小
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)

        # 亚像素级边缘插值
        edge_points = np.argwhere(edges > 0)
        subpixel_edges = []
        for (y, x) in edge_points:
            # 获取周围像素的梯度大小
            g1 = magnitude[y, x]
            g2 = magnitude[y, x+1] if x+1 < magnitude.shape[1] else g1  # 避免越界
            
            # 亚像素插值位置计算
            subpixel_x = x + (g1 - g2) / (g1 + g2) if (g1 + g2) != 0 else x
            subpixel_edges.append((y, subpixel_x))

        # 绘制亚像素边缘点
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for (y, subpixel_x) in subpixel_edges:
            cv2.circle(output_image, (int(subpixel_x), y), 1, (0, 255, 0), -1)

        # 保存处理后的图像到输出文件夹
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, output_image)
        print(f"已处理并保存图像: {output_path}")


# import cv2
# import numpy as np

# # 输入图像路径
# input_image_path = '/data1/home_data/chendi/datasets/grinding/images/train/Image_20241007175054023.bmp'  # 修改为您的图像路径
# output_image_path = '/data1/home_data/chendi/datasets/grinding/images/preproposs/01.bmp'  # 保存处理后的图像路径

# # 读取图像
# image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# # 检查图像是否读取成功
# if image is None:
#     raise FileNotFoundError(f"无法读取图像文件: {input_image_path}")

# # 预处理：高斯滤波去噪
# blurred = cv2.GaussianBlur(image, (5, 5), 0)

# # 使用Canny边缘检测
# edges = cv2.Canny(blurred, 100, 200)

# # 获取边缘的梯度方向和梯度大小
# sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
# sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
# magnitude = np.sqrt(sobelx**2 + sobely**2)
# direction = np.arctan2(sobely, sobelx)

# # 亚像素级边缘插值
# edge_points = np.argwhere(edges > 0)
# subpixel_edges = []
# for (y, x) in edge_points:
#     # 获取周围像素的梯度大小
#     g1 = magnitude[y, x]
#     g2 = magnitude[y, x+1] if x+1 < magnitude.shape[1] else g1  # 避免越界
    
#     # 亚像素插值位置计算
#     subpixel_x = x + (g1 - g2) / (g1 + g2) if (g1 + g2) != 0 else x
#     subpixel_edges.append((y, subpixel_x))

# # 绘制亚像素边缘点
# output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# for (y, subpixel_x) in subpixel_edges:
#     cv2.circle(output_image, (int(subpixel_x), y), 1, (0, 255, 0), -1)

# # 保存处理后的图像
# cv2.imwrite(output_image_path, output_image)
# print(f"已处理并保存图像: {output_image_path}")