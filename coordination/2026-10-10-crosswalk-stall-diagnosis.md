# 2026-10-10 新巡航会话在横向标记前持续停车：只读诊断

用户反馈“怎么这次直接就卡死了”。读取最新session `0b2c18d8d3374d68ad2bad33a709357c`，run `20261010_043922_911806_hardware`。总时限0已生效，结束原因为interrupted by operator，stopped_verified=true；没有程序异常或推理错误，219个模型结果已完成。助手未重启服务、发启动/停止请求、改控制代码或模型。

CSV共276行：264stop/12forward，最后一次运动sample94/timestamp24.7481。sample95开始出现 `fitted centerline leaves semantic corridor`；最后100行全部是此估计失效。末100行53次visual-current-evidence-veto、47次执行前过期，后者覆盖了用户看到的主停车reason，但estimate_reason保留了实际几何失效。推理中位98.722ms，执行年龄中位0.2973s；时效问题亦存在，但不是100帧无可行路径的唯一原因。

纯CPU读取sample94/95/170/275的原始消费frame、road、lane，复算保存原始掩码形成的当前corridor（不调用GPU模型、相机或硬件）。94有效，其余拟合检查失败且当前排除掩码没有可达近到远路径；不是此前“全局多项式切弯但仍有曲线路径”的情况，不能直接复用该修补。

sample275原始camera图可见前方斑马线/横向白色路面标记。YOLO road头在mask行148:182、列96:256的支持比例为1.0；lane头在同一区域支持.22684，其中独立横向component为(x49,y175,width221,height8,area1110)，另一个为(x107,y154,width93,height5,area339)。车端将全部lane像素从road扣除，再只保留ego分支；当前ego通道因此仅保留近底部区域，无法到达固定lookahead行148。行179图像中心road=1、lane=1，整行lane228像素。视觉照片及掩码支持“模型把横向路面标记纳入lane头、统一排除使通路切断”的解释，不表示road头本身准确性已全面验证。

证据目录 `outputs/visual_feedback/20261010/path-stall-fix/`：diagnosis.json、四个stored-input overlay、corridor.npz、path-mask.png、275-raw-heads.png、provenance.json。均为派生副本，不改原始采集。

尚未修复横向路面标记与不可跨越边界的区分；不能简单删全部横线或恢复全部排除像素，否则可能放行黄色前边界/封闭区域。下一修改需要以当前模型road支持、当前原始图像中的标记形态及边界证据区分，再做少量存储输入对照；过期保护继续保留。当前无总时限和直道新观察续期实现仍有效，但本轮不能当作实际连续驾驶或过弯成功。
