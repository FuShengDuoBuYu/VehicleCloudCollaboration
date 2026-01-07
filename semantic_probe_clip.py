import argparse
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

DEFAULT_LABELS = [
    # 正常道路语义（做对照用）
    "normal road driving scene",
    "ordinary street scene",
    "highway driving scene",
    "tunnel driving scene",
    "intersection",

    # 交通参与者
    "pedestrian",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "scooter",

    # 临时施工/长尾要素
    "traffic cone",
    "traffic bollard",
    "road barrier",
    "barricade",
    "construction sign",
    "detour sign",
    "lane closed sign",
    "temporary traffic sign",
    "warning sign",
    "arrow board",
    "road work",

    # 信号/标志（你可继续扩展）
    "traffic light",
    "stop sign",
    "speed limit sign",
    "yield sign",
    "no entry sign",
]

def softmax(x: torch.Tensor):
    x = x - x.max()
    return torch.softmax(x, dim=-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="image path")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS), help="comma separated text labels")
    ap.add_argument("--longtail_mode", choices=["simple", "entropy"], default="entropy",
                    help="simple: max score < thr; entropy: distribution too flat")
    ap.add_argument("--thr", type=float, default=0.28, help="threshold for simple mode")
    args = ap.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    img = Image.open(args.img).convert("RGB")

    # 轻量、通用、CPU能跑：openai/clip-vit-base-patch32
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    inputs = processor(text=labels, images=img, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0].float()  # (num_labels,)

    probs = softmax(logits)

    # 排序输出 TopK
    topk = min(args.topk, len(labels))
    vals, idx = torch.topk(probs, k=topk)
    print("\n=== Semantic Probe (CLIP) Top-{} ===".format(topk))
    for p, i in zip(vals.tolist(), idx.tolist()):
        print(f"{labels[i]:<28}  {p:.3f}")

    # ---- 长尾判定（你现在先用一个简单可解释的）----
    # simple: 最高概率太低 -> “没有一个概念很像”
    # entropy: 分布太平 -> “模型也拿不准”
    max_p = probs.max().item()
    eps = 1e-8
    entropy = float(-(probs * (probs + eps).log()).sum().item())  # nats
    norm_entropy = entropy / (torch.log(torch.tensor(len(labels))).item())  # 0~1

    if args.longtail_mode == "simple":
        is_long_tail = (max_p < args.thr)
        reason = f"max_prob={max_p:.3f} < thr={args.thr:.3f}"
    else:
        # 一般经验：norm_entropy 高说明分布很“平”，模型不确定
        is_long_tail = (norm_entropy > 0.72) or (max_p < 0.22)
        reason = f"norm_entropy={norm_entropy:.3f}, max_prob={max_p:.3f}"

    print("\n=== Long-tail Determination ===")
    print("is_long_tail:", bool(is_long_tail))
    print("reason:", reason)

if __name__ == "__main__":
    main()
