from modelscope import snapshot_download

model_dir = snapshot_download(
    'OpenBMB/MiniCPM-V-2_6',
    cache_dir='./MiniCPM'   # 下载到当前目录的 MiniCPM 文件夹下
)
print(model_dir)