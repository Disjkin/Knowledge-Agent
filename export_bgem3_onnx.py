"""把已下载的 bge-m3 pytorch_model.bin 导出为 ONNX 模型。

背景：pytorch_model.bin (2.27GB) 已通过 ModelScope 完整下载，
但 CPU 版 PyTorch 跑 bge-m3 太慢。ONNX Runtime 推理快 5-10 倍。
本脚本用 CPU torch 加载模型并一次性导出 ONNX，之后嵌入用 onnxruntime。
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

MODEL_DIR = os.path.expanduser(
    "~/.cache/modelscope/models/BAAI--bge-m3/snapshots/master"
)
OUT_DIR = "C:/tmp/bge-m3-onnx"  # 纯 ASCII 路径，避免中文路径导致 ONNX 导出失败


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"模型目录: {MODEL_DIR}", flush=True)
    print(f"导出到: {OUT_DIR}", flush=True)

    import torch
    from transformers import AutoModel, AutoTokenizer

    print("加载 bge-m3 (CPU, 约 1-2 分钟)...", flush=True)
    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model.eval()
    print(f"加载完成 ({time.time()-t0:.0f}s)", flush=True)

    # 保存 tokenizer 到导出目录
    tokenizer.save_pretrained(OUT_DIR)

    # 构造示例输入
    dummy = tokenizer(
        ["测试文本"], padding=True, truncation=True, max_length=64, return_tensors="pt"
    )
    dummy_inputs = (
        dummy["input_ids"],
        dummy["attention_mask"],
    )

    # 只导出 dense 编码部分：用 XLM-RoBERTa 的 last_hidden_state + CLS pooling
    class DenseEncoder(torch.nn.Module):
        """取 [CLS] token 输出作为 dense 向量。"""

        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, input_ids, attention_mask):
            outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # [B, L, H]
            # [CLS] token 是第一位（XLM-RoBERTa 结构）
            dense = last_hidden[:, 0]  # [B, H]
            # L2 归一化（bge-m3 dense 向量是归一化的）
            dense = torch.nn.functional.normalize(dense, p=2, dim=1)
            return dense

    encoder = DenseEncoder(model)

    print("导出 ONNX...", flush=True)
    t1 = time.time()
    torch.onnx.export(
        encoder,
        dummy_inputs,
        os.path.join(OUT_DIR, "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["dense_vecs"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "dense_vecs": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # 传统导出器，不依赖 onnxscript
    )
    print(f"ONNX 导出完成 ({time.time()-t1:.0f}s)", flush=True)

    # 验证
    import onnxruntime as ort

    sess = ort.InferenceSession(os.path.join(OUT_DIR, "model.onnx"))
    inputs = {
        "input_ids": dummy["input_ids"].numpy(),
        "attention_mask": dummy["attention_mask"].numpy(),
    }
    out = sess.run(["dense_vecs"], inputs)[0]
    print(f"验证: dense_vecs shape={out.shape}, norm={out[0].sum()**2:.4f}", flush=True)
    print(f"\n导出完成! 文件: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
