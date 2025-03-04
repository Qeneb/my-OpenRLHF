# download_model.py
import os
from huggingface_hub import snapshot_download
from pathlib import Path

def download_model(
    model_name: str = "Qwen/Qwen2-0.5B",
    local_dir: str = "./models/qwen2-0.5b",
    ignore_patterns: list = ["*.msgpack", "*.h5", "*.ot", "*.tflite"],
    proxy: str = None,
    mirror: str = None
):
    """
    从HuggingFace下载模型到本地
    参数说明：
    - model_name: HuggingFace模型ID
    - local_dir: 本地保存路径
    - ignore_patterns: 忽略的文件模式（排除非必要的大文件）
    - proxy: 代理地址，例如 "http://127.0.0.1:7890"
    - mirror: 镜像地址，例如 "https://hf-mirror.com"
    """
    # 设置环境变量
    if proxy:
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy
    if mirror:
        os.environ["HF_ENDPOINT"] = mirror

    # 创建保存目录
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    # 执行下载
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            ignore_patterns=ignore_patterns,
            max_workers=4                 # 并行下载线程数
        )
        print(f"\n\033[32m✅ 模型已成功下载到 {local_dir}\033[0m")
    except Exception as e:
        print(f"\n\033[31m❌ 下载失败: {str(e)}\033[0m")
        return

    # 验证关键文件
    required_files = [
        "config.json",
        "model.safetensors",  # 或 pytorch_model.bin
        "tokenizer.json",
        "generation_config.json"
    ]
    
    print("\n正在验证文件完整性...")
    missing_files = []
    for file in required_files:
        path = os.path.join(local_dir, file)
        if not os.path.exists(path):
            missing_files.append(file)
    
    if missing_files:
        print(f"\033[33m⚠️ 缺失关键文件: {', '.join(missing_files)}\033[0m")
    else:
        print("\033[32m✅ 所有关键文件完整\033[0m")

if __name__ == "__main__":
    # 使用示例（根据需要修改参数）
    download_model(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        local_dir="./models/qwen2.5-1.5b-instruct",
        # proxy="http://127.0.0.1:7890",  # 如果需要代理
        mirror="https://hf-mirror.com"   # 如果使用镜像
    )