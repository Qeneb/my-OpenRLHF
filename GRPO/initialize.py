# !pip install tf-keras # for some reason, Hugging Face cannot work without it
# !pip install flash-attn # FlashAttention2
# !pip install wandb # Weights and Biases
# !pip install 'accelerate>=0.26.0'
# !pip install transformers # Hugging Face Transformers API
# !pip install datasets # Hugging Face Datasets API

# Install necessary libraries
def install_libraries():
    import subprocess

    packages = [
        "tf-keras",
        "flash-attn",
        "wandb",
        "accelerate>=0.26.0",
        "transformers",
        "datasets"
    ]

    for package in packages:
        try:
            command = ["pip", "install", package, '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple']
            subprocess.run(command, check=True)
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")

# Import necessary libraries
# Basic Python libraries for various operations
import random
import numpy as np
import wandb
import os

# PyTorch and related libraries for deep learning
import torch

def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None

    Explanation:
        1. Sets seed for Python's built-in random module for basic random operations.
        2. Sets seed for NumPy, ensuring consistent random number generation in array operations.
        3. Sets seed for PyTorch CPU operations.
        4. If CUDA is available, sets seed for all GPU devices.
        5. Configures cuDNN to ensure deterministic behavior:
           - Sets deterministic flag to True, ensuring reproducible results.
           - Disables benchmarking to prevent algorithm selection based on hardware.

    Note:
        Setting deterministic behavior may impact performance but ensures consistent results
        across multiple runs, which is crucial for debugging and research.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call the function to set random seed for reproducibility
# set_random_seed(42)

def check_and_set_wandb(project_name="GRPO-Qwen-1.5-Instruct-Multi-GPU"):
    # 获取当前登录账号信息
    user_info = wandb.api.viewer()
    print(f"当前wandb账号：{user_info['username']}")

    # 初始化 API
    api = wandb.Api()
    # 获取当前用户的实体名称
    entity = api.viewer.entity

    # 检查项目是否存在
    project_exists = False
    try:
        # 尝试获取指定项目
        api.project(name=project_name, entity=entity)
        project_exists = True
    except wandb.CommError:
        # 如果项目不存在，会抛出 CommError 异常
        project_exists = False

    if project_exists:
        print(f"项目 {project_name} 已存在。")
    else:
        print(f"项目 {project_name} 不存在，正在创建...")
        # 初始化一个新的 wandb 运行，指定项目名称
        run = wandb.init(project=project_name)
        # 完成运行，避免不必要的资源占用
        run.finish()
        print(f"项目 {project_name} 已成功创建。")
    # # Set environment variables for Weights & Biases (wandb) logging
    # os.environ["WANDB_API_KEY"] = "USE YOUR KEY"
    os.environ["WANDB_PROJECT"] = project_name

if __name__ == '__main__':
    check_and_set_wandb()