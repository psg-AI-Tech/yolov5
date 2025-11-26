import onnxruntime as ort
import numpy as np
import cv2

def preprocess_image(image_path, input_shape=(640, 640), mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]):
    """
    修复返回值问题：明确返回 2 个结果（输入张量 + 解码所需参数）
    YoloV5 标准预处理：letterbox 填充 + 归一化 + float32 转换
    """
    # 1. 读取图像（BGR → RGB）
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")  # 增加图片读取校验
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]  # 原始图像尺寸（高、宽）
    h, w = input_shape      # 模型输入尺寸（640,640）

    # 2. Letterbox 填充：保持长宽比，避免图像拉伸（YoloV5 必需）
    scale = min(w / w0, h / h0)  # 缩放比例（取最小，防止超出输入尺寸）
    new_w, new_h = int(w0 * scale), int(h0 * scale)  # 缩放后的图像尺寸
    pad_w, pad_h = (w - new_w) // 2, (h - new_h) // 2  # 左右、上下填充量

    # 缩放图像 + 填充灰边（YoloV5 默认填充色 114）
    img_resized = cv2.resize(img, (new_w, new_h))
    img_padded = np.full((h, w, 3), 114, dtype=np.uint8)  # 初始化填充图像
    img_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w, :] = img_resized  # 放入缩放后的图像

    # 3. 数据预处理：归一化 + 转 float32（解决之前的类型报错）
    img_array = img_padded.astype(np.float32)  # 强制转为 float32
    img_array = img_array / 255.0              # 归一化到 [0, 1]
    # 确保 mean/std 是 float32，避免自动转 double
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    img_normalized = (img_array - mean) / std

    # 4. 调整维度：HWC → CHW → 加 batch 维度（匹配模型输入 [1,3,640,640]）
    img_transposed = np.transpose(img_normalized, (2, 0, 1))  # (3,640,640)
    input_tensor = np.expand_dims(img_transposed, axis=0)     # (1,3,640,640)

    # 5. 返回 2 个结果：输入张量 + 原始尺寸/填充信息（后续解码边界框需要）
    return input_tensor, (h0, w0, scale, pad_w, pad_h)

def postprocess_output(output_tensor, h0, w0, scale, pad_w, pad_h, conf_threshold=0.5, iou_threshold=0.45):
    """
    修正后的后处理函数，修复 NMSBoxes 参数错误
    """
    outputs = output_tensor[0]  # 去掉 batch 维度 (25200,7)
    
    # 1. 筛选置信度 > 阈值的框
    obj_conf = outputs[:, 4]
    valid_mask = obj_conf > conf_threshold
    valid_boxes = outputs[valid_mask]

    if len(valid_boxes) == 0:
        return []

    # 2. 转换边界框格式：(cx, cy, w, h) → (x1, y1, x2, y2)
    cx, cy = valid_boxes[:, 0], valid_boxes[:, 1]
    w, h = valid_boxes[:, 2], valid_boxes[:, 3]
    x1 = (cx - w/2 - pad_w) / scale
    y1 = (cy - h/2 - pad_h) / scale
    x2 = (cx + w/2 - pad_w) / scale
    y2 = (cy + h/2 - pad_h) / scale

    # 3. 限制坐标在图像范围内
    x1 = x1.clip(0, w0)
    y1 = y1.clip(0, h0)
    x2 = x2.clip(0, w0)
    y2 = y2.clip(0, h0)

    # 4. 计算综合置信度
    cls_conf = valid_boxes[:, 5]
    conf = obj_conf[valid_mask] * cls_conf
    cls_id = np.zeros_like(cls_conf, dtype=np.int32)

    # 5. 准备 NMS 输入（关键修正点）
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    confidences = conf.astype(np.float32)
    
    # 6. 执行 NMS（注意参数顺序和类型）
    try:
        # OpenCV 4.6+ 的正确调用方式
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),       # 必须转换为 list[list[float]]
            scores=confidences.tolist(), # 必须转换为 list[float]
            score_threshold=conf_threshold,
            nms_threshold=iou_threshold
        )
    except Exception as e:
        print(f"NMS 执行错误: {e}")
        return []

    # 7. 整理最终结果
    final_boxes = []
    if len(indices) > 0:
        for idx in indices.flatten():
            final_boxes.append([
                boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3],
                confidences[idx], cls_id[idx]
            ])
    return np.array(final_boxes)

def infer_onnx_model(onnx_path, image_path):
    """
    完整推理流程：加载模型 → 预处理 → 推理 → 后处理 → 输出结果
    """
    # 1. 加载 ONNX 模型（CPU 推理，GPU 需修改 providers）
    providers = ["CPUExecutionProvider"]
    # providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]  # GPU 加速（需安装 onnxruntime-gpu）
    try:
        net = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        raise RuntimeError(f"加载模型失败：{e}")

    # 2. 获取模型输入/输出节点（已确认：input=images，output=output0）
    input_name = net.get_inputs()[0].name
    output_name = net.get_outputs()[0].name
    print(f"模型输入：名称={input_name}, 形状={net.get_inputs()[0].shape}")
    print(f"模型输出：名称={output_name}, 形状={net.get_outputs()[0].shape}")

    # 3. 图像预处理（返回 2 个值，解决 unpack 报错）
    input_tensor, (h0, w0, scale, pad_w, pad_h) = preprocess_image(
        image_path=image_path,
        input_shape=(640, 640),  # 匹配模型输入尺寸
        mean=[0.0, 0.0, 0.0],    # 若训练时用 (img/255-0.5)/0.5，改为 mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
        std=[1.0, 1.0, 1.0]
    )
    print(f"输入张量：类型={input_tensor.dtype}，形状={input_tensor.shape}")  # 验证：float32 + (1,3,640,640)

    # 4. 执行推理（无类型报错）
    try:
        outputs = net.run([output_name], {input_name: input_tensor})
    except Exception as e:
        raise RuntimeError(f"推理失败：{e}")
    output_tensor = outputs[0]

    # 5. 输出后处理（解码边界框）
    detected_boxes = postprocess_output(output_tensor, h0, w0, scale, pad_w, pad_h)

    # 6. 打印检测结果
    print("\n===== 检测结果 =====")
    if len(detected_boxes) == 0:
        print("未检测到人脸")
    else:
        for i, box in enumerate(detected_boxes):
            x1, y1, x2, y2, conf, cls_id = box
            print(f"人脸 {i+1}：坐标=({x1:.1f},{y1:.1f})~({x2:.1f},{y2:.1f})，置信度={conf:.4f}，类别ID={cls_id}")

    return detected_boxes



if __name__ == "__main__":
    onnx_model_path = "/home/jl/_github/f_python/yolov5/runs/train/exp/weights/test_face.onnx"  # 你的 ONNX 模型路径
    test_image_path = "/mnt/e/data_set_self/YOLO_Mask/score/images/test/val_1.jpg"   # 测试图片路径（人脸图片）
    try:
        result = infer_onnx_model(onnx_model_path, test_image_path)
        img = cv2.imread(test_image_path)
        for box in result:
            x1, y1, x2, y2, conf, cls_id = map(int, box[:6])
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, f"{conf:.2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        
    except Exception as e:
        print(f"程序出错：{e}")
    
    # 后续处理（根据模型任务扩展）：
    # - 人脸识别：result 是人脸特征向量，可用于计算余弦相似度匹配
    # - 人脸检测：result 可能是边界框 + 置信度，需解码为 (x1,y1,x2,y2,score)
    # - 表情识别：result 是类别概率，用 np.argmax(result) 取预测类别