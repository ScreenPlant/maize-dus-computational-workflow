import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def predict_single_image(model, image_path, save_path=None, mask_save_path=None, show=False):
    """对单张图像进行预测"""
    # 加载图像
    image = cv2.imread(image_path)
    # 确保图像已加载
    if image is None:
        print(f"无法加载图像: {image_path}")
        return

    # 模型预测
    results = model.predict(image, conf=0.5, iou=0.7)

    # 获取预测结果
    result = results[0]

    # 可视化结果（带边界框）
    annotated = result.plot()

    # 保存带边界框的结果
    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"带边界框的预测结果已保存至: {save_path}")

    # 生成并保存只有分割区域的图像
    if mask_save_path:
        # 创建全黑图像
        mask_image = np.zeros_like(image)

        # 如果有分割掩码
        if result.masks is not None:
            # 为每个掩码提取原图区域
            for i, mask in enumerate(result.masks.data.cpu().numpy()):
                # 调整掩码大小以匹配原图
                mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))

                # 将掩码应用到原图，提取该区域
                masked_region = cv2.bitwise_and(image, image, mask=mask)

                # 将该区域添加到结果图像
                mask_image = cv2.bitwise_or(mask_image, masked_region)

        # 保存掩码图像
        cv2.imwrite(mask_save_path, mask_image)
        print(f"分割区域图像已保存至: {mask_save_path}")

    # 显示结果
    if show:
        plt.figure(figsize=(15, 10))

        plt.subplot(1, 2, 1)
        plt.title('带边界框的预测')
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        if mask_save_path and os.path.exists(mask_save_path):
            mask_result = cv2.imread(mask_save_path)
            plt.subplot(1, 2, 2)
            plt.title('分割区域')
            plt.imshow(cv2.cvtColor(mask_result, cv2.COLOR_BGR2RGB))
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    return results


def predict_folder(model, folder_path, save_dir=None, mask_save_dir=None, show=False):
    """对文件夹中的所有图像进行预测"""
    # 创建保存目录
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if mask_save_dir and not os.path.exists(mask_save_dir):
        os.makedirs(mask_save_dir)

    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(folder_path)
                   if os.path.isfile(os.path.join(folder_path, f))
                   and os.path.splitext(f)[1].lower() in image_extensions]

    # 对每张图像进行预测
    all_results = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        if save_dir:
            save_path = os.path.join(save_dir, image_file)
        else:
            save_path = None

        if mask_save_dir:
            # 确保掩码图像保存为JPG格式
            base_name, _ = os.path.splitext(image_file)
            mask_save_path = os.path.join(mask_save_dir, f"{base_name}.jpg")
        else:
            mask_save_path = None

        print(f"正在处理图像: {image_file}")
        results = predict_single_image(model, image_path, save_path, mask_save_path, show)
        all_results.append(results)

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLO segmentation inference on a single image or a folder.")
    parser.add_argument("--model", required=True, help="Path to a YOLO model checkpoint.")
    parser.add_argument("--mode", choices=["single", "folder"], default="single", help="Inference mode.")
    parser.add_argument("--input", required=True, help="Input image path or input folder path.")
    parser.add_argument("--save-path", default=None, help="Annotated output image path for single-image mode.")
    parser.add_argument("--mask-save-path", default=None, help="Mask-only output image path for single-image mode.")
    parser.add_argument("--save-dir", default=None, help="Annotated output directory for folder mode.")
    parser.add_argument("--mask-save-dir", default=None, help="Mask-only output directory for folder mode.")
    parser.add_argument("--show", action="store_true", help="Display results.")
    args = parser.parse_args()

    model = YOLO(args.model)
    if args.mode == 'single':
        predict_single_image(model, args.input, args.save_path, args.mask_save_path, show=args.show)
    else:
        predict_folder(model, args.input, args.save_dir, args.mask_save_dir, show=args.show)
