"""用 onnxruntime quantization 把 bge-m3 ONNX 模型量化到 INT8。

速度提升 2-3 倍，内存减半。量化一次，之后推理用 INT8 模型。
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

MODEL_DIR = "C:/tmp/bge-m3-onnx"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
INT8_PATH = os.path.join(MODEL_DIR, "model_int8.onnx")


def main():
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if not os.path.exists(MODEL_PATH):
        print(f"模型不存在: {MODEL_PATH}", flush=True)
        sys.exit(1)

    print(f"量化模型: {MODEL_PATH}", flush=True)
    t0 = time.time()
    quantize_dynamic(
        model_input=MODEL_PATH,
        model_output=INT8_PATH,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Attention", "Gemm", "LayerNormalization"],
        per_channel=True,
    )
    print(f"INT8 量化完成 ({time.time()-t0:.0f}s): {INT8_PATH}", flush=True)
    print(f"文件大小: {os.path.getsize(INT8_PATH)/1024/1024:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
