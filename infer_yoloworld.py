from ultralytics import YOLO

img = "/home/fushengduobuyu/VehicleCloudCollaboration/000002_1616005520000.jpg"
out_dir = "/home/fushengduobuyu/VehicleCloudCollaboration/runs_world"

# 最强（CPU 会慢）：yolov8x-worldv2.pt
model = YOLO("yolov8x-worldv2.pt")

# 你的“交通参与者 + 特殊路标/临时设施”
classes = [
    "person","car","truck","bus","motorcycle","bicycle","scooter",
    "traffic cone","traffic bollard","road barrier","traffic barrier","barricade",
    "construction sign","road work sign","detour sign","lane closed sign","temporary sign","arrow board","warning sign",
    "traffic light","stop sign","speed limit sign","pedestrian crossing sign","no entry sign","yield sign",
    ""
]

model.set_classes(classes)

results = model.predict(
    source=img,
    imgsz=640,
    conf=0.20,
    save=True,
    project=out_dir,
    name="pred",
)
print("saved to:", results[0].save_dir)
