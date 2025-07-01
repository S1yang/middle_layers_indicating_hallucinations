import torch
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

def debug_original_llava(model_path):
    """
    Load the original llava model and print its key configurations for debugging.
    """
    # Load model components
    load_8bit = False
    load_4bit = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name,
        load_8bit, load_4bit, device=device
    )
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # 1. 打印模型配置
    print("=== Model Configuration ===")
    for k, v in model.config.to_dict().items():
        print(f"{k}: {v}")
    
    # 2. 如果有 generation_config，也打印
    if hasattr(model, "generation_config"):
        print("\n=== Generation Configuration ===")
        for k, v in model.generation_config.to_dict().items():
            print(f"{k}: {v}")
    
    # 3. 打印图像处理器配置
    print("\n=== Image Processor Configuration ===")
    print(image_processor)
    
    # 4. 打印分词器关键信息
    print("\n=== Tokenizer Configuration ===")
    print(f"Model max length: {tokenizer.model_max_length}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    # 查看初始化参数
    if hasattr(tokenizer, "init_kwargs"):
        print(f"Tokenizer init kwargs: {tokenizer.init_kwargs}")
    
    # 5. 打印 context length
    print(f"\nContext length (max token length): {context_len}")

# 使用示例
if __name__ == "__main__":
    model_path = "liuhaotian/llava-v1.5-7b"  # 替换为实际路径或模型标识
    debug_original_llava(model_path)