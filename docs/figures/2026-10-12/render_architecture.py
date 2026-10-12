"""Render source-grounded thesis diagrams; no vehicle imports or network access.

Run: MPLCONFIGDIR=/tmp/vcc-architecture-mpl python3 -B render_architecture.py
Outputs: PNG previews and SVG beside this file. PDF is excluded by user request.
Evidence mapping: coordination/2026-10-12-repository-review.md.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

OUT = Path(__file__).resolve().parent
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
# SVG uses portable glyph outlines.
plt.rcParams.update({'svg.fonttype': 'path'})
INK = '#173047'
BLUE = '#267398'
PURPLE = '#765391'
GREEN = '#39796C'
GRAY = '#768794'
RED = '#A75251'


def canvas(height):
    fig, ax = plt.subplots(figsize=(16, height / 100), dpi=150)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 1600), ylim=(height, 0))
    ax.axis('off')
    fig.set_facecolor('white')
    return fig, ax


def text(ax, x, y, value, size=12, color=INK, ha='left'):
    return ax.text(x, y, value, fontproperties=FONT, fontsize=size,
                   color=color, ha=ha, va='center', linespacing=1.65, zorder=5)


def panel(ax, x, y, w, h, title, face, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=14',
                              linewidth=1, facecolor=face, edgecolor=edge, zorder=0))
    text(ax, x + 22, y + 32, title, 15, edge)


def box(ax, x, y, w, h, title, body, color=BLUE, face='white'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=9',
                              linewidth=1.1, facecolor=face, edgecolor=color, zorder=3))
    text(ax, x + 17, y + 28, title, 13, color)
    if body:
        text(ax, x + 17, y + (h + 35) / 2, body, 10.5)


def arrow(ax, points, color=BLUE, dashed=False):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=1.45,
                ls=(0, (5, 3)) if dashed else '-', zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                 mutation_scale=13, linewidth=1.45, color=color,
                 linestyle=(0, (5, 3)) if dashed else '-', zorder=2))


def save(fig, stem):
    for ext in ('png', 'svg'):
        fig.savefig(OUT / (stem + '.' + ext), dpi=180, facecolor='white')
    plt.close(fig)


def detailed():
    fig, ax = canvas(1320)
    text(ax, 40, 46, '车云协同感知、决策与受限执行架构', 23)
    text(ax, 42, 86, '当前实现视图  ·  Jetson / Rosmaster  ·  视觉反馈与云辅助恢复', 12, GRAY)
    text(ax, 1555, 47, '2026-10-12  |  6647934', 10, GRAY, 'right')
    panel(ax, 35, 125, 1530, 90, '操作者与运行管理', '#F4F6F8', GRAY)
    text(ax, 275, 169, '面板启停  →  会话监督与设备占用协调     ·     心跳 / 人工结束     ·     状态展示与历史回放', 12)
    panel(ax, 35, 250, 1120, 770, '车端  /  当前观测驱动的本地闭环', '#F1F7FB', BLUE)
    panel(ax, 1190, 250, 375, 770, '云端  /  外部模型服务', '#F7F3FA', PURPLE)

    box(ax, 70, 330, 255, 110, 'RGB 相机', '原始画面与捕获时钟\n帧序号、图像方向')
    box(ax, 380, 330, 315, 110, '异步 YOLOPv2 感知', '道路 / 车道线 / 目标检测\nCUDA · FP16 · TorchScript')
    box(ax, 750, 330, 360, 110, '当前局部证据', '同帧掩码、黄线与路面支持\n可见通道 / 路径 / 障碍约束')
    arrow(ax, [(325, 385), (380, 385)])
    arrow(ax, [(695, 385), (750, 385)])

    box(ax, 70, 520, 255, 130, '底盘遥测', 'IMU：短转角度与时效约束\n编码器 / UART：记录与复盘\n不等同于完整里程计导航')
    box(ax, 380, 520, 315, 130, '本地路径与控制提案', 'LCC 差速巡航\n视觉反馈短步 / 右转\n独立本地许可与截止时间')
    box(ax, 750, 520, 360, 130, '停车事件与异步云客户端', '持续停车确认，构造受限候选\n1–3 张原图 + 事件 / 帧上下文\n单个在途请求，超时与尝试预算', PURPLE)
    arrow(ax, [(850, 440), (850, 475), (537, 475), (537, 520)])
    arrow(ax, [(1020, 440), (1020, 520)], PURPLE, True)
    text(ax, 1040, 481, '当前证据', 9, PURPLE)
    arrow(ax, [(325, 585), (380, 585)])
    arrow(ax, [(695, 585), (750, 585)], PURPLE, True)

    box(ax, 1220, 330, 315, 130, '多模态场景理解', '外部百炼服务\nQwen Realtime\n模型配置见图注', PURPLE)
    box(ax, 1220, 520, 315, 130, '图像与场景上下文', '独立异步请求\n语义解释 / 风险识别\n从车端候选中选择或保持停车', PURPLE)
    arrow(ax, [(1110, 585), (1220, 585)], PURPLE, True)
    text(ax, 1163, 552, '上传', 9, PURPLE, 'center')
    arrow(ax, [(1377, 520), (1377, 460)], PURPLE, True)
    box(ax, 1220, 760, 315, 190, '结构化回复', 'road-recovery-v1\n场景判断 / 危险 / 原因\n候选 ID 或 hold\n\n不返回 PWM、任意轨迹或时长', PURPLE)
    arrow(ax, [(1518, 460), (1550, 460), (1550, 718), (1377, 718), (1377, 760)], PURPLE, True)

    box(ax, 70, 760, 255, 130, '雷达与深度', 'ROS 采集、原始记录与展示\n记录健康参与运动准入\n未接入当前几何避障 / SLAM', GRAY)
    box(ax, 380, 760, 315, 130, '车端决策融合与复核', '统一 FusionDecision / 规则策略\n事件、合同、候选与新帧复核\n云建议仅可授权一个有界短步')
    box(ax, 750, 760, 360, 130, '最终执行安全门', '图像 / 语义 / IMU 时效\n黄线与障碍、记录健康\n动作截止 / 看门狗 / 人工停止', RED)
    arrow(ax, [(537, 650), (537, 760)])
    text(ax, 555, 717, '本地提案', 9, BLUE)
    arrow(ax, [(1220, 855), (1175, 855), (1175, 697), (650, 697), (650, 760)], PURPLE, True)
    text(ax, 900, 678, '异步建议返回车端复核', 10, PURPLE, 'center')
    arrow(ax, [(695, 825), (750, 825)])
    arrow(ax, [(325, 860), (350, 860), (350, 917), (820, 917), (820, 890)], GRAY, True)
    text(ax, 550, 905, '记录健康', 9, GRAY)
    box(ax, 750, 940, 360, 65, '车型适配 → 串口 → 四轮执行', '', BLUE)
    arrow(ax, [(980, 890), (980, 940)])
    text(ax, 748, 1038, '软件约束：PWM ≤ 30；云短步 ≤ 0.15 s / 5°，执行后重新观察', 10, BLUE, 'center')
    arrow(ax, [(1090, 1005), (1090, 1060), (48, 1060), (48, 400), (70, 400)], BLUE)
    text(ax, 400, 1043, '车辆与环境反馈 → 新一轮相机观测', 10, BLUE, 'center')

    # Operator commands remain a distinct control plane.
    arrow(ax, [(1140, 215), (1140, 792), (1110, 792)], RED, True)

    panel(ax, 35, 1100, 1530, 160, '实验数据与复现链', '#F0F7F4', GREEN)
    text(ax, 70, 1185, '原始证据\n视频 / 掩码 / UART / 雷达 / 深度 / 云事件', 11, GREEN)
    text(ax, 640, 1185, '运行关联与冻结\nrun ID · 配置 · 源码 / 模型版本 · 时钟', 11, GREEN)
    text(ax, 1180, 1185, '离线回放与派生分析\n指标 / 失败复盘 / 论文图表', 11, GREEN)
    arrow(ax, [(515, 1185), (610, 1185)], GREEN)
    arrow(ax, [(1055, 1185), (1150, 1185)], GREEN)
    text(ax, 790, 1240, '各运行模块共同留证；派生分析不覆盖原始数据；回放不产生新的物理轨迹证据', 10, GREEN, 'center')
    text(ax, 40, 1290, '实线：本地观测 / 控制    紫色虚线：云请求 / 建议    红色虚线：操作者停止约束    灰色虚线：记录健康', 10, GRAY)
    save(fig, 'current-architecture')


def overview():
    fig, ax = canvas(610)
    text(ax, 40, 45, '云端语义辅助、车端受限执行的协同闭环', 22)
    text(ax, 40, 86, '当前方法概览  ·  持续停车触发云辅助恢复  ·  2026-10-12 / 6647934', 11, GRAY)
    panel(ax, 35, 265, 1530, 235, '车端本地闭环', '#F1F7FB', BLUE)
    box(ax, 1220, 130, 315, 100, '云端多模态理解', '场景解释 / 风险 / 候选选择', PURPLE)
    positions = [70, 375, 680, 985, 1290]
    labels = [('当前感知', '相机 + YOLOPv2\n道路 / 边界 / 障碍'),
              ('本地决策', '可见路径与 LCC\n视觉反馈控制'),
              ('运行状态与触发', '正常提案直接通过\n持续停车时请求云端'),
              ('融合复核', '新帧 / 事件 / 候选\n统一车端规则仲裁'),
              ('受限执行', '安全门与截止时间\n底盘执行后再观察')]
    for x, (title, body) in zip(positions, labels):
        box(ax, x, 335, 240, 120, title, body)
    for left, right in zip(positions[:-1], positions[1:]):
        arrow(ax, [(left + 240, 395), (right, 395)])
    arrow(ax, [(800, 335), (800, 180), (1220, 180)], PURPLE, True)
    text(ax, 985, 155, '异步上传原图 / 上下文 / 候选', 10, PURPLE, 'center')
    arrow(ax, [(1377, 230), (1377, 245), (1105, 245), (1105, 335)], PURPLE, True)
    text(ax, 1105, 222, '结构化建议', 10, PURPLE, 'center')
    arrow(ax, [(1510, 455), (1510, 480), (90, 480), (90, 455)])
    text(ax, 800, 535, '原始观测、控制输出、云请求 / 回复与运行版本共同归档 → 离线回放与论文分析', 12, GREEN, 'center')
    text(ax, 800, 582, '云端不直接驱动电机；时序掩码突变分支未在当前 profile 启用；本图不表示正式实车性能已验证', 10, GRAY, 'center')
    save(fig, 'method-overview')


if __name__ == '__main__':
    detailed()
    overview()
    print('Rendered current-architecture and method-overview as PNG / SVG.')
