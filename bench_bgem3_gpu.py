"""Benchmark: bge-m3 ONNX 在 CPU vs DirectML(GPU) 上的实测对比。

不依赖 app 代码，直接加载 C:/tmp/bge-m3-onnx 下的 ONNX 模型，
对四种组合各跑一遍编码计时：
  - INT8 / CPUExecutionProvider      （当前生产用的组合）
  - INT8 / DmlExecutionProvider      （作者注释说慢 6 倍，复现确认）
  - FP32 / CPUExecutionProvider
  - FP32 / DmlExecutionProvider      （未测过，重点验证）

batch=1 模拟单次检索 query，batch=8 模拟入库批量嵌入。
Run: .venv/Scripts/python.exe bench_bgem3_gpu.py
"""
import gc
import os
import time

import numpy as np

MODEL_DIR = "C:/tmp/bge-m3-onnx"
INT8_PATH = os.path.join(MODEL_DIR, "model_int8.onnx")   # 1.36GB 自包含
FP32_PATH = os.path.join(MODEL_DIR, "model.onnx")        # 1.3MB 图 + 外部权重

# 强制离线，避免 tokenizer 加载时联网校验卡住
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ~960 字的中文样本文本（≈ 600 token，tokenizer 会截断到 512，接近真实 1000 字 chunk）
SAMPLE = "这是一个用于测试中文嵌入模型吞吐量的样本文本，涉及知识库检索与语义匹配。" * 20


def make_session(model_path: str, provider: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(4, os.cpu_count() or 4)
    # INT8 模型若开图优化会卡 10-30 分钟，必须禁用；FP32 用 BASIC 即可
    is_int8 = "int8" in os.path.basename(model_path).lower()
    opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL if is_int8
        else ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    providers = [provider, "CPUExecutionProvider"]
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)


def encode(session, tok, texts, max_length=512):
    enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="np")
    inputs = {}
    for name in [i.name for i in session.get_inputs()]:
        if name == "input_ids":
            inputs[name] = enc["input_ids"].astype(np.int64)
        elif name == "attention_mask":
            inputs[name] = enc["attention_mask"].astype(np.int64)
    out = session.run([o.name for o in session.get_outputs()], inputs)
    return out[0]


def bench_session(label, model_path, provider, tok, rounds=3):
    """加载一次 session，连测 batch=1 和 batch=8，测完释放。"""
    try:
        sess = make_session(model_path, provider)
    except Exception as e:
        print(f"  [{label:18s}] 加载失败: {e}")
        return {}
    out = {}
    for batch_size in (1, 8):
        texts = [SAMPLE + str(i) for i in range(batch_size)]
        try:
            encode(sess, tok, texts[:1])  # warmup
            times = []
            for _ in range(rounds):
                t0 = time.perf_counter()
                encode(sess, tok, texts)
                times.append(time.perf_counter() - t0)
            avg = sum(times) / len(times)
            out[batch_size] = avg
            print(f"  [{label:18s}] batch={batch_size}  avg={avg*1000:7.1f}ms  "
                  f"({batch_size/avg:6.1f} texts/s)  runs={[round(t*1000) for t in times]}ms")
        except Exception as e:
            print(f"  [{label:18s}] batch={batch_size} 失败: {e}")
    del sess
    gc.collect()
    return out


def main():
    from onnxruntime import get_available_providers
    print("onnxruntime providers:", get_available_providers())
    print(f"INT8 模型存在: {os.path.exists(INT8_PATH)}   FP32 模型存在: {os.path.exists(FP32_PATH)}")
    print()
    tok = load_tokenizer()

    combos = [("INT8", INT8_PATH), ("FP32", FP32_PATH)]
    providers = ["CPUExecutionProvider", "DmlExecutionProvider"]

    results = {}
    for mp_label, mp in combos:
        for prov in providers:
            key = f"{mp_label}/{prov.split('Execution')[0]}"
            print(f"--- {key} ---")
            results[key] = bench_session(key, mp, prov, tok)
        print()

    print("=== 结论（加速比 = INT8/CPU 耗时 ÷ 该组合耗时，>1 表示比当前生产快）===")
    base_b1 = results.get("INT8/CPU", {}).get(1)
    base_b8 = results.get("INT8/CPU", {}).get(8)
    for key, by_batch in results.items():
        b1, b8 = by_batch.get(1), by_batch.get(8)
        s1 = f"{b1*1000:7.1f}ms" if b1 else "    N/A"
        s8 = f"{b8*1000:7.1f}ms" if b8 else "    N/A"
        ratio = f"{base_b1/b1:.2f}x" if (base_b1 and b1) else "-"
        print(f"  {key:18s}  batch1={s1}  batch8={s8}  速度比={ratio}")


if __name__ == "__main__":
    main()
