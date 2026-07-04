import argparse
import json
import os
from typing import Dict, List, Tuple


def normalize_polygon(points: List[List[float]], img_width: int, img_height: int) -> List[float]:
    """将多边形顶点坐标归一化到[0,1]范围"""
    normalized = []
    for x, y in points:
        normalized.extend([x / img_width, y / img_height])
    return normalized


def labelme_json_to_yolo_seg(json_path: str, output_dir: str, class_mapping: Dict[str, int]) -> None:
    """将LabelMe的JSON标注文件转换为YOLO分割格式的TXT文件"""
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_width = data.get('imageWidth')
        img_height = data.get('imageHeight')

        if not img_width or not img_height:
            print(f"警告: {json_path} 缺少图片尺寸信息，跳过")
            return

        txt_filename = os.path.splitext(os.path.basename(json_path))[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, 'w', encoding='utf-8') as f:
            for shape in data.get('shapes', []):
                label = shape.get('label')
                if label not in class_mapping:
                    print(f"警告: {json_path} 中存在未映射的类别 '{label}'，跳过")
                    continue

                points = shape.get('points')
                if not points or len(points) < 3:
                    print(f"警告: {json_path} 中 '{label}' 的多边形顶点不足，跳过")
                    continue

                normalized_points = normalize_polygon(points, img_width, img_height)
                class_id = class_mapping[label]

                # 写入格式: class_id x1 y1 x2 y2 ... xn yn
                line = f"{class_id} " + " ".join([f"{p:.6f}" for p in normalized_points]) + "\n"
                f.write(line)

        print(f"成功转换: {json_path} -> {txt_path}")

    except Exception as e:
        print(f"错误: 处理 {json_path} 时出错 - {str(e)}")


def batch_convert_labelme(json_dir: str, output_dir: str, class_mapping: Dict[str, int]) -> None:
    """批量转换目录下所有LabelMe的JSON文件"""
    for filename in os.listdir(json_dir):
        if filename.lower().endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            labelme_json_to_yolo_seg(json_path, output_dir, class_mapping)


def parse_class_mapping(values: List[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid class mapping '{item}'. Expected format label=id.")
        label, class_id = item.split("=", 1)
        mapping[label.strip()] = int(class_id.strip())
    return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LabelMe polygons to YOLO segmentation labels.")
    parser.add_argument("--json-dir", required=True, help="Directory containing LabelMe JSON files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for YOLO segmentation labels.")
    parser.add_argument(
        "--class-map",
        nargs="+",
        required=True,
        help="Class mapping entries in the format label=id, e.g. leaf=0 tassel=1.",
    )
    args = parser.parse_args()

    class_mapping = parse_class_mapping(args.class_map)
    batch_convert_labelme(args.json_dir, args.output_dir, class_mapping)
