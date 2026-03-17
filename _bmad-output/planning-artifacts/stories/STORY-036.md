# STORY-036: 实现 ONNXEmbedder Provider

**Epic:** EPIC-007 (本地 Embedding + Daemon)
**Priority:** Should Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 3

---

## User Story

As a privacy-first user
I want local embedding without internet
So that my data never leaves my machine

---

## Description

### Background
Memorus 的 L4 VectorSearcher 依赖 Embedding 向量进行语义搜索。默认情况下需要调用 OpenAI 等在线 API，这与 Memorus 的"数据本地化"原则相悖。ONNXEmbedder 使用 ONNX Runtime 在本地运行 all-MiniLM-L6-v2 模型，实现零网络依赖的 Embedding 能力。需注册到 mem0 的 EmbedderFactory，使用 `provider="onnx"` 即可启用。

### Scope
**In scope:**
- ONNXEmbedder 实现 mem0 `EmbeddingBase` 接口
- 模型文件自动下载到 `~/.memorus/models/`
- Tokenization (使用 tokenizers 库)
- ONNX Runtime inference
- 注册到 mem0 EmbedderFactory
- batch embed 支持

**Out of scope:**
- 模型微调/训练
- 多模型切换（未来考虑 paraphrase-multilingual-MiniLM）
- GPU 加速（默认 CPU）

---

## Acceptance Criteria

- [ ] `ONNXEmbedder` 实现 `embed(text) -> list[float]` 接口
- [ ] 默认模型 `all-MiniLM-L6-v2`，输出 384 维向量
- [ ] 模型文件自动下载到 `~/.memorus/models/`（首次使用时）
- [ ] 模型已下载后零网络请求（离线可用）
- [ ] `embed_batch(texts) -> list[list[float]]` 批量 Embedding
- [ ] 单条 embed < 10ms（CPU，文本 < 200 字符）
- [ ] 注册到 mem0 EmbedderFactory，`provider="onnx"` 可用
- [ ] 模型文件损坏时自动重新下载
- [ ] onnxruntime 未安装时 `import memorus` 不报错，仅在使用 ONNX 时报 ImportError

---

## Technical Notes

### Components
- `memorus/embeddings/__init__.py` — 包入口
- `memorus/embeddings/onnx.py` — ONNXEmbedder

### API Design

```python
from mem0.embeddings.base import EmbeddingBase

class ONNXEmbedderConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    dimensions: int = 384
    model_dir: str = "~/.memorus/models/"
    auto_download: bool = True
    max_length: int = 256  # token limit

class ONNXEmbedder(EmbeddingBase):
    def __init__(self, config: ONNXEmbedderConfig | None = None):
        self._config = config or ONNXEmbedderConfig()
        self._session: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        self._ensure_loaded()
        tokens = self._tokenize(text)
        outputs = self._session.run(None, tokens)
        return self._mean_pooling(outputs).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]  # simple loop, batch later

    def _ensure_loaded(self) -> None:
        if self._session is None:
            model_path = self._resolve_model_path()
            if not model_path.exists():
                self._download_model()
            self._session = ort.InferenceSession(str(model_path))
            self._tokenizer = Tokenizer.from_pretrained(self._config.model)

    def _mean_pooling(self, outputs) -> np.ndarray: ...
    def _tokenize(self, text: str) -> dict: ...
    def _download_model(self) -> None: ...
    def _resolve_model_path(self) -> Path: ...
```

### Model Download Strategy
1. 检查 `~/.memorus/models/all-MiniLM-L6-v2/model.onnx` 是否存在
2. 不存在 → 从 Hugging Face Hub 下载（`huggingface_hub.hf_hub_download`）
3. 下载失败 → 抛 `RuntimeError("Failed to download ONNX model")`
4. 文件校验：加载模型失败 → 删除并重新下载

### Dependencies
- `onnxruntime` — ONNX 推理（optional dependency）
- `tokenizers` — 文本 tokenization
- `huggingface_hub` — 模型下载
- `numpy` — 向量操作

### Dependencies on Existing Code
- `memorus/config.py:MemorusConfig` — 可能需要添加 ONNXEmbedderConfig
- mem0 `EmbeddingBase` — embed() 接口

### Edge Cases
- 文本为空 → 返回零向量
- 文本超长（>256 tokens）→ 截断到 max_length
- 磁盘空间不足 → 下载失败，抛出明确错误
- 并发调用 embed → ONNX Runtime 内部线程安全
- Windows 路径包含中文 → Path 对象处理

---

## Dependencies

**Prerequisite Stories:**
- STORY-003: MemorusConfig ✓（已完成）

**Blocked Stories:**
- STORY-037: MemorusDaemon 服务端（预加载 ONNX 模型）

---

## Definition of Done

- [ ] `memorus/embeddings/onnx.py` 实现 ONNXEmbedder
- [ ] 注册到 mem0 EmbedderFactory
- [ ] 单元测试（mock ONNX session）
- [ ] 集成测试（真实模型，可选，需网络下载）
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **ONNXEmbedder 核心逻辑:** 2 points
- **模型下载 + 文件管理:** 1.5 points
- **EmbedderFactory 注册:** 0.5 points
- **测试:** 1 point
- **Total:** 5 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
