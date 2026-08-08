"""ONNX Runtime 嵌入后端 - 用 ONNX 版 bge-m3 跑嵌入。

为什么用 ONNX：
- bge-m3 的 PyTorch 版在 CPU 上极慢（568M 参数 FP32）
- ONNX Runtime 有深度优化，且支持 DirectML 走 GPU

GPU 加速（DirectML）—— RTX 2060 实测（见 bench_bgem3_gpu.py）：
  INT8/CPU  1255ms/batch1   （原生产配置，最慢）
  INT8/DML   191ms/batch1   ≈ 6.5x
  FP32/DML   157ms/batch1   ≈ 8.0x  ← 默认
- DmlExecutionProvider 不可用时自动回落 CPU。
- FP32 在 GPU 上比 INT8 还快：DirectML 对量化算子加速不佳，INT8 部分算子
  会回落 CPU；故默认走 FP32/DML。通过 settings.local_embedding_int8 切换。

模型文件位于 C:/tmp/bge-m3-onnx（export_bgem3_onnx.py 导出、quantize_bgem3.py 量化）：
  FP32 -> model.onnx + 外部权重文件（base.embeddings.* / onnx__MatMul_*）
  INT8 -> model_int8.onnx（自包含）
"""
import logging
import os

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ONNX 模型目录（纯 ASCII 路径，中文路径会导致导出/加载问题）
_MODEL_DIR = "C:/tmp/bge-m3-onnx"


def _pick_providers() -> list[str]:
    """优先 DirectML(GPU)，不可用则回落 CPU。"""
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]
    if "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class BgeM3OnnxEmbeddings:
    """基于 ONNX Runtime 的 bge-m3 嵌入模型（实现 LangChain Embeddings 接口）。"""

    def __init__(self, model_dir: str = _MODEL_DIR, use_int8: bool | None = None):
        import onnxruntime as ort

        # use_int8 未指定时读 settings：True=INT8 量化版，False=FP32
        if use_int8 is None:
            use_int8 = getattr(settings, "local_embedding_int8", False)

        model_file = "model_int8.onnx" if use_int8 else "model.onnx"
        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            # 指定精度不存在时互相回退
            alt = "model.onnx" if use_int8 else "model_int8.onnx"
            alt_path = os.path.join(model_dir, alt)
            if os.path.exists(alt_path):
                logger.warning(f"ONNX 模型 {model_file} 不存在，回退到 {alt}")
                model_path = alt_path
                use_int8 = not use_int8
            else:
                raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")

        # 配置 ONNX Runtime
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(4, os.cpu_count() or 4)
        # INT8 已量化优化过，开图优化会卡 10-30 分钟 -> 禁用；FP32 用 BASIC
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL if use_int8
            else ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )

        providers = _pick_providers()
        self._session = ort.InferenceSession(
            model_path, sess_options=opts, providers=providers
        )
        self._tokenizer = self._load_tokenizer(model_dir)

        # 模型输入/输出名
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        device = "GPU(DirectML)" if "DmlExecutionProvider" in providers else "CPU"
        logger.info(
            f"ONNX bge-m3 加载完成: model={'INT8' if use_int8 else 'FP32'} "
            f"device={device} inputs={self._input_names} outputs={self._output_names}"
        )

    @staticmethod
    def _load_tokenizer(model_dir: str):
        """加载 bge-m3 的 tokenizer（从导出目录加载）。"""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 分批编码（每批 8 句，避免显存/内存峰值）
        all_vecs: list[np.ndarray] = []
        BATCH = 8
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            all_vecs.append(self._encode(batch))
        return np.concatenate(all_vecs, axis=0).tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self._encode([text])[0]
        return vec.tolist()

    def _encode(self, texts: list[str]) -> np.ndarray:
        """对文本列表编码，返回归一化嵌入矩阵。"""
        # 批量 tokenize（padding 到最长）
        # max_length=512: bge-m3 dense 检索推荐长度，1000 字 chunk 约 600 token，
        # 截断尾部 ~15% 对检索影响极小，但推理速度快 4 倍
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        # 组织 ONNX 输入（导出模型只有 input_ids 和 attention_mask）
        inputs = {}
        for name in self._input_names:
            if name == "input_ids":
                inputs[name] = encoded["input_ids"].astype(np.int64)
            elif name == "attention_mask":
                inputs[name] = encoded["attention_mask"].astype(np.int64)

        outputs = self._session.run(self._output_names, inputs)

        # 取 dense 向量（导出模型只有一个输出 dense_vecs）
        dense = outputs[0]

        # 归一化（导出时已归一化，这里保险再归一次）
        dense = np.asarray(dense, dtype=np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return dense / norms


def get_onnx_embeddings():
    """工厂函数：返回 ONNX 嵌入实例。"""
    return BgeM3OnnxEmbeddings()
