# 2026-10-10 弯道路线起点、近处掩码与绘图耗时

用户确认结束后要求继续。session `5406fc4d3c394761b186b992d4cde82d`，run `20261010_050650_512765_hardware`：operator clicked end、exit0、stopped_verified=true，2354控制行且2354归档帧，dropped0/errorNone。不是网页退出。全部原始日志/视频/NPZ/传感器记录保持不变。本轮不发送运动请求、不开相机/UART、不重启服务、不安装/提交。

## 根因与代码修复

路线从图像底边239行初始化，路径规划从225行（bottom_ratio=.94）出发。底边road碎片使路线先选右侧x230+分支，向前裁掉中心附近真实当前road；即使sample700/1000原始掩码有路径，裁剪后也没有起点。仅visual模式改为从规划器同一起始行选择支路，保留其下方原始像素供侵蚀/逐行检查；live构建和recorded回放由centerline.bottom_ratio派生path_start_ratio。旧模式保持原选择。样例1728原始掩码本就没有足够的侵蚀后起点，继续停车，不能靠这项修复制造道路。

仅起点修改的原始输入配对130行：正常窗口有效13/20不变、最终动作不变；首次卡住15/40→23/40；后段4/40→7/40；更后段5/30→10/30。后两段最终仍全stop，明确不足以宣称解决弯道。保留原始捕获/消费时间、IMU、外部及最终执行否决，窗口控制器重置。这揭示近处road缺口与恢复/应用时效仍是实际瓶颈。

## 输入分辨率数据调整

在已结束本轮的原始帧上离线重新运行相同TorchScript权重、FP16、完整道路/车道/目标头，不打开实车传感器。最初受限环境CUDA初始化失败；GPU读取通过授权主机命令完成，不安装依赖。320与384的四个帧对照：单帧路径3/4→4/4。随后384在690–709、1720–1739两段40个采样（33个唯一归档输入）有39个单帧路径有效；启用原时序滤波后38/40有效。采用旧时钟/执行否决的控制回放仍39stop/1turn-right；不是新的实车轨迹或端到端实时验证。

四帧384推理81.26–102.98ms；较长探查中位166.366ms，最大698.159ms，后台monitor可能争用资源、初始编译也影响波动。不能拿四帧宣传恒定10Hz，也不能把旧时钟回放当成384新延迟模拟。仅当前visual profile输入由320改为384，输出掩码仍320×240、目标头仍开启；其他profile不改。这是针对近处道路细节的试跑选择，实际频率和应用过期仍待新run记录。

## 绘图耗时

原渲染对宽道路做NumPy布尔索引收集/回写，会延迟下一次控制/模型输入。替换为恒定绿色层与OpenCV copyTo，仅复制相同visible掩码；红色优先、文字/路径/输出保持。五张派生图片各4次CPU对照：20次输出逐像素一致，绘图中位28.802→14.610ms。不是实时GPU或端到端耗时保证。

## 验证与加载

最后执行：`outputs/field_yolopv2/venv/bin/python -m unittest car.test.test_semantic_bend_path car.test.test_track_colors car.test.test_yolopv2_primary car.test.test_control_render_timing car.test.test_outer_loop_replay car.test.test_stationary_corner_onboard.StationaryCalibrationTests.test_main_logs_native_encoder_snapshot_once_per_control_frame -q`，55项通过12.486s；合成CPU/临时视频/虚拟驱动，不访问实车。底边误选测试先失败后通过；真缺失起点仍无效。git diff检查通过。独立只读审查起点/live-replay一致性及后加384/渲染均无important问题。

effective config只读确认img_size384、mask240×320、camera20fps、source最大年龄.3s、bottom_ratio.94、颜色与continuous开启。原始road排除/黄线/障碍、IMU、启动恢复门禁、局部截止与取消锁存均保留，总时限仍0。下一次用户Start由新进程加载，不需要重新加载supervisor；仅空闲后台monitor可能还在旧配置上。

本轮证据 `outputs/visual_feedback/20261010/corner-target-fix/`：diagnosis、before/after、incremental.patch、paired-records/replay-summary、model-resolution-probe、model-sequence-probe、size384-inputs、size384-controller-records/summary、render-benchmark、source/input/model hash、verification。HEAD `fce828b89b084eda293e83b99c8861f522858d8e`及大量preexisting dirty保留，无提交。下一实车重点看实际模型/应用年龄、近处road有效率、连续直道与是否出现合规短步/pivot；实际过弯未验证，不承诺此次一定通过。
