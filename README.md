# VehicleCloudCollaboration

仓库当前按职责拆分为两层：

- cloud: 云端推理服务
- car: 车端代码

在 car 内再按功能隔离为：

- longtail: 长尾检测与评估逻辑
- control: DonkeyCar 车控相关逻辑

## 目录结构

```text
VehicleCloudCollaboration/
├── cloud/
└── car/
    ├── longtail/
    └── control/
```

## car/longtail

包含原有长尾检测相关代码与数据：

- main.py
- classifier.py
- benchmark.py
- evaluate.py
- config.yaml
- requirements.txt
- detectors/
- dataset/

运行示例：

```bash
cd /home/car/Desktop/VehicleCloudCollaboration/car/longtail
pip install -r requirements.txt
python main.py
```

## car/control

包含原有 DonkeyCar 与底层车控代码：

- donkeycar/
- mycar/
- utils/

例如 DonkeyCar 管理入口仍在：

```bash
cd /home/car/Desktop/VehicleCloudCollaboration/car/control/mycar
python manage.py drive
```

## 说明

本次仅做代码整理与分层隔离，不新增架构图里尚未落地的新功能。
