from datasets import load_dataset

dataset_name = "OpenRLHF/preference_dataset_mixture2_and_safe_pku"
data_dir = None

# 下载并缓存数据集（默认保存到~/.cache/huggingface/datasets）
dataset = load_dataset(dataset_name, data_dir=data_dir)  

# 保存到本地目录
dataset.save_to_disk("./datasets/"+dataset_name) #+"@"+data_dir)
