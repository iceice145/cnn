#coding:utf-8
#根据实际情况更换模型
# yolov11n.yaml (nano)：轻量化模型，适合嵌入式设备，速度快但精度略低。
# yolov11s.yaml (small)：小模型，适合实时任务。
# yolov11m.yaml (medium)：中等大小模型，兼顾速度和精度。
# yolov11b.yaml (base)：基本版模型，适合大部分应用场景。
# yolov11l.yaml (large)：大型模型，适合对精度要求高的任务。

from ultralytics import YOLO

model_path = r"C:\Users\lsh\Desktop\jyjc\YOLOv11晶圆体缺陷检测系统\ultralytics\cfg\models\11\yolo11-myCBAM+C3CA.yaml"
data_path = 'data.yaml'

if __name__ == '__main__':
    zxmodel = YOLO(model_path)
    results = zxmodel.train(data=data_path,
                          epochs=100,
                          batch=16,
                          device=0,
                          workers=0,
                          project='runs',
                          name='exp',
                          )








