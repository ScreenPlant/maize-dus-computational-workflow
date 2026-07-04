from ultralytics import YOLO

# 加载训练好的最佳分割模型
model = YOLO("runs/seg/exp17/weights/best.pt")

# 导出为ONNX格式（参数与命令行对应）
success = model.export(
    format="onnx",    # 目标格式
    imgsz=1000,        # 输入尺寸
    simplify=True,    # 简化模型
    half=True,        # FP16半精度
    # dynamic=True,   # 可选：动态输入尺寸，取消注释启用
)

print("ONNX导出成功" if success else "ONNX导出失败")