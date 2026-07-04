import argparse
import os
import json
import numpy as np
import cv2
from PIL import Image  # 新增PIL库用于图像读取


def process_labelme_folder(input_folder, output_folder, suffix="_mask"):
    os.makedirs(output_folder, exist_ok=True)
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    print(f"找到 {len(json_files)} 个json文件")

    for json_file in json_files:
        json_path = os.path.join(input_folder, json_file)
        print(f"\n处理文件: {json_file}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            image_path = data.get('imagePath')
            if not image_path:
                print(f"警告: 缺少imagePath，跳过")
                continue

            # 构建可能的图像路径
            potential_image_paths = [
                os.path.join(input_folder, image_path),
                os.path.splitext(json_path)[0] + '.jpg',
                os.path.splitext(json_path)[0] + '.png',
            ]

            # 查找有效图像文件
            image_file = None
            for path in potential_image_paths:
                normalized_path = os.path.normpath(path)
                if os.path.exists(normalized_path):
                    image_file = normalized_path
                    break

            if not image_file:
                print(f"警告: 未找到对应图像")
                continue

            # 关键修改：用PIL读取图像，再转换为OpenCV格式
            try:
                # 使用PIL读取（支持更多格式）
                pil_img = Image.open(image_file)
                # 转换为RGB格式（避免Alpha通道问题）
                if pil_img.mode in ('RGBA', 'LA'):
                    background = Image.new(pil_img.mode[:-1], pil_img.size, (255, 255, 255))
                    background.paste(pil_img, pil_img.split()[-1])
                    pil_img = background
                elif pil_img.mode == 'P':
                    pil_img = pil_img.convert('RGB')
                # 转换为OpenCV的BGR格式
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                print(f"成功加载图像: {os.path.basename(image_file)}")
            except Exception as e:
                print(f"警告: 无法读取图像 {image_file}: {str(e)}")
                continue

            # 处理标注并生成mask（后续步骤不变）
            for i, shape in enumerate(data.get('shapes', [])):
                shape_type = shape.get('shape_type', '')
                if shape_type not in ['polygon', 'rectangle']:
                    continue

                points = np.array(shape['points'], dtype=np.int32)
                mask = np.zeros_like(image, dtype=np.uint8)
                cv2.fillPoly(mask, [points], (255, 255, 255))
                masked_image = cv2.bitwise_and(image, mask)

                base_name = os.path.splitext(os.path.basename(json_file))[0]
                output_name = f"{base_name}_{i}{suffix}.jpg" if len(data['shapes']) > 1 else f"{base_name}{suffix}.jpg"
                output_path = os.path.join(output_folder, output_name)
                cv2.imwrite(output_path, masked_image)
                print(f"已保存: {output_path}")

        except Exception as e:
            print(f"错误: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate masked crops from LabelMe annotations.")
    parser.add_argument("--input-folder", required=True, help="Directory containing LabelMe JSON files and images.")
    parser.add_argument("--output-folder", required=True, help="Directory for masked output images.")
    parser.add_argument("--suffix", default="_mask", help="Suffix appended to output image names.")
    args = parser.parse_args()

    process_labelme_folder(args.input_folder, args.output_folder, suffix=args.suffix)
    print("处理完成!")
