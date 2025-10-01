import os
import numpy as np
import cv2

# 文件夹路径
data_folder = 'runs/predict/base/labels'  # 替换为你的文件夹路径
image_folder = '/data1/home_data/chendi/datasets/grinding/images/val'  # 图像所在文件夹路径

# 最小二乘法拟合直线
def fit_line(points):
    x = points[:, 0]
    y = points[:, 1]
    # 最小二乘法拟合直线 y = mx + b
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b

# 最小二乘法拟合圆
def fit_circle(points):
    x = points[:, 0]
    y = points[:, 1]
    # 构建线性系统
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x**2 + y**2
    # 最小二乘法求解
    c, d, e = np.linalg.lstsq(A, b, rcond=None)[0]
    cx = c
    cy = d
    radius = np.sqrt(e + cx**2 + cy**2)
    return (cx, cy), radius

# 读取所有 .txt 文件
for file_name in os.listdir(data_folder):
    if file_name.endswith('.txt'):
        txt_path = os.path.join(data_folder, file_name)
        
        # 读取数据
        with open(txt_path, 'r') as file:
            data = list(map(float, file.readline().strip().split()))

        # 提取点（从索引 5 开始）
        points = np.array(data[5:]).reshape(-1, 3)  # 每个点有三个值（x, y, confidence）

        # 缩放点坐标
        scale_x = 4096
        scale_y = 3000
        points[:, 0] *= scale_x  # x 坐标乘以 4096
        points[:, 1] *= scale_y  # y 坐标乘以 3000

        # 使用前四个点拟合两条直线
        line1_points = points[:2, :2]  # 前两个点 (x, y)
        line2_points = points[2:4, :2]  # 第三和第四个点 (x, y)

        # 最小二乘法拟合直线
        m1, b1 = fit_line(line1_points)
        m2, b2 = fit_line(line2_points)

        # 计算夹角（以弧度为单位）
        def calculate_angle(m1, m2):
            if m1 == np.inf:  # 垂直线的情况
                return np.pi / 2 if m2 != 0 else 0
            if m2 == np.inf:
                return np.pi / 2 if m1 != 0 else 0
            angle_rad = np.arctan(abs((m2 - m1) / (1 + m1 * m2)))
            return angle_rad

        angle = calculate_angle(m1, m2)  # 夹角
        angle_deg = np.degrees(angle)  # 转换为度

        # 拟合圆（使用第5、6、7个点）
        circle_points = points[4:7, :2]  # 第5、6、7个点 (x, y)
        center, radius = fit_circle(circle_points)

        # 读取对应的图片
        img_name = file_name.replace('.txt', '.bmp')  # 假设图片为 .jpg 格式
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)

        # 绘制直线和圆
        # 将直线1绘制到图像上
        x_vals = np.array([0, img.shape[1]])
        y_vals1 = m1 * x_vals + b1
        y_vals2 = m2 * x_vals + b2
        cv2.line(img, (int(x_vals[0]), int(y_vals1[0])), (int(x_vals[1]), int(y_vals1[1])), (0, 255, 0), 2)
        cv2.line(img, (int(x_vals[0]), int(y_vals2[0])), (int(x_vals[1]), int(y_vals2[1])), (0, 255, 0), 2)

        if center is not None:
            # 调整圆心坐标为整数，并绘制圆
            center_int = tuple(np.round(center).astype(int))
            radius_int = int(np.round(radius))
            cv2.circle(img, center_int, radius_int, (255, 0, 0), 2)  # 绘制圆形

        # 缩放半径
        radius_scaled = radius * 0.0008625  # 仅在输出时缩放半径

        # 显示夹角和缩放后的半径
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0  # 设置字体大小为2.0
        thickness = 2  # 设置线宽为2

        # 文本的 y 坐标
        text_y_offset = 1350

        # 显示夹角
        cv2.putText(img, f'Angle: {angle_deg:.1f} deg', (1200, text_y_offset), font, font_scale, (0, 0, 255), thickness)

        # 显示半径
        text_y_offset += 80  # 设置第二行文本的间距
        cv2.putText(img, f'Radius: {radius_scaled:.2f} mm', (1200, text_y_offset), font, font_scale, (0, 0, 255), thickness)

        # 保存处理后的图像
        output_path = os.path.join(image_folder, 'output', img_name)  # 输出路径
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)

        print(f'Processed {img_name}, Angle: {angle_deg:.2f}, Radius: {radius_scaled:.2f}')

print("所有文件处理完成！")
