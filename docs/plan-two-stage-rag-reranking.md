# Two-Stage RAG with Reranking Integration Plan

## Executive Summary

本計劃描述如何將 **Qwen3-Reranker** 引入 Codebase RAG MCP Server，實現兩階段檢索（Two-Stage Retrieval）架構，以提升搜尋結果的準確度和相關性。

### 預期效益
- 搜尋準確度提升 **22-31%**（根據 Milvus 實測數據）
- 更精確的 function-level 程式碼檢索
- 支援 instruction-aware reranking，可針對不同查詢類型優化

---

## 1. 架構概述

### 1.1 現有架構（Single-Stage）

```
Query → Embedding → Vector Search → Top-N Results → Return to User
                         ↓
                   Qdrant (ANN)
```

**限制：**
- Bi-encoder 只能獨立編碼 query 和 document，無法捕捉深度交互
- ANN 搜尋效率高但語義精度有限
- 相同 score 的結果無法進一步區分品質

### 1.2 新架構（Two-Stage with Reranking）

```
Query → Embedding → Stage 1: Vector Search → Top-K Candidates
                           ↓
                      Qdrant (ANN)
                           ↓
              Stage 2: Cross-Encoder Reranking
                           ↓
                   Reranked Top-N Results → Return to User
```

**優勢：**
- Stage 1 快速篩選大量候選（K = 50-100）
- Stage 2 精確重排序（Cross-encoder 深度語義分析）
- 結合效率與精度

---

## 2. Reranker Model 選擇

### 2.1 推薦：Qwen3-Reranker

| Model | 參數量 | 延遲 (CPU) | 延遲 (GPU) | MTEB Score |
|-------|--------|------------|------------|------------|
| Qwen3-Reranker-0.6B | 0.6B | ~380ms | ~85ms | 高 |
| Qwen3-Reranker-4B | 4B | ~1.2s | ~180ms | 更高 |
| Qwen3-Reranker-8B | 8B | ~2.5s | ~300ms | 最高 |

**建議選擇：Qwen3-Reranker-0.6B**
- Apple Silicon (M1/M2/M3) 優化良好
- 延遲可接受（本機部署 ~300-400ms）
- 可透過 Ollama 或直接載入 transformers

### 2.2 本機部署選項

**選項 A：Ollama 部署（推薦）**
```bash
ollama pull qwen3:reranker
```

**選項 B：MLX Server 部署（Apple Silicon 優化）**
- 利用現有 MLX embedding server 架構擴展

**選項 C：Transformers 直接載入**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
reranker = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B")
```

---

## 3. 實作計劃

### Phase 1: Reranker Service 核心實作

#### 3.1 新增 `src/services/reranker_service.py`

```python
"""
Reranker service for two-stage RAG retrieval.
Implements cross-encoder reranking using Qwen3-Reranker.
"""

from dataclasses import dataclass
from typing import Any
import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class RerankedResult:
    """Reranked search result with updated relevance score."""
    content: str
    file_path: str
    original_score: float
    rerank_score: float
    metadata: dict[str, Any]

class RerankerService:
    """Cross-encoder reranking service."""

    PROVIDER_TRANSFORMERS = "transformers"
    PROVIDER_OLLAMA = "ollama"
    PROVIDER_MLX = "mlx"

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.provider = os.getenv("RERANKER_PROVIDER", self.PROVIDER_TRANSFORMERS)
        self.model_name = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
        self.max_length = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
        self.batch_size = int(os.getenv("RERANKER_BATCH_SIZE", "8"))

        self._model = None
        self._tokenizer = None
        self._token_true_id = None
        self._token_false_id = None

    def _load_model(self):
        """Lazy load the reranker model."""
        if self._model is not None:
            return

        self.logger.info(f"Loading reranker model: {self.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side='left'
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name
        ).eval()

        # Get yes/no token IDs for relevance scoring
        self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")
        self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")

        # Move to MPS if available (Apple Silicon)
        if torch.backends.mps.is_available():
            self._model = self._model.to("mps")
            self.logger.info("Reranker model loaded on MPS (Apple Silicon)")

    def compute_relevance_score(
        self,
        query: str,
        document: str,
        instruction: str | None = None
    ) -> float:
        """
        Compute relevance score for a query-document pair.

        Args:
            query: Search query
            document: Document content to score
            instruction: Optional task instruction for instruction-aware reranking

        Returns:
            Relevance score between 0 and 1
        """
        self._load_model()

        # Build input with optional instruction
        if instruction:
            prompt = f"<instruct>{instruction}</instruct>\n<query>{query}</query>\n<doc>{document}</doc>"
        else:
            prompt = f"<query>{query}</query>\n<doc>{document}</doc>"

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )

        if torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}

        # Get logits
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits[:, -1, :]

        # Calculate relevance score from yes/no logits
        yes_logit = logits[:, self._token_true_id]
        no_logit = logits[:, self._token_false_id]

        # Softmax to get probability
        score = torch.softmax(torch.stack([no_logit, yes_logit], dim=-1), dim=-1)
        relevance_score = score[:, 1].item()  # Probability of "yes"

        return relevance_score

    def rerank_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int | None = None,
        instruction: str | None = None
    ) -> list[RerankedResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Original search query
            results: List of search results from Stage 1
            top_k: Number of top results to return (default: all)
            instruction: Optional task instruction

        Returns:
            List of reranked results sorted by relevance
        """
        if not results:
            return []

        self._load_model()

        reranked = []

        # Process in batches for efficiency
        for result in results:
            content = result.get("content", "")
            if not content:
                continue

            # Compute rerank score
            rerank_score = self.compute_relevance_score(
                query=query,
                document=content,
                instruction=instruction
            )

            reranked.append(RerankedResult(
                content=content,
                file_path=result.get("file_path", ""),
                original_score=result.get("score", 0.0),
                rerank_score=rerank_score,
                metadata={
                    "chunk_type": result.get("chunk_type", ""),
                    "name": result.get("name", ""),
                    "line_start": result.get("line_start", 0),
                    "line_end": result.get("line_end", 0),
                    "language": result.get("language", ""),
                    "breadcrumb": result.get("breadcrumb", ""),
                }
            ))

        # Sort by rerank score (descending)
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked
```

### Phase 2: 整合到 Search Pipeline

#### 3.2 修改 `src/tools/indexing/search_tools.py`

在 `search_sync()` 函數中加入 reranking 階段：

```python
# 新增參數
def search_sync(
    query: str,
    n_results: int = 5,
    cross_project: bool = False,
    search_mode: str = "hybrid",
    include_context: bool = True,
    context_chunks: int = 1,
    target_projects: list[str] | None = None,
    collection_types: list[str] | None = None,
    # 新增 reranking 參數
    enable_reranking: bool = True,       # 是否啟用 reranking
    rerank_top_k: int = 50,              # Stage 1 候選數量
    rerank_instruction: str | None = None,  # 可選的任務指令
) -> dict[str, Any]:
    """..."""

    # Stage 1: 原有的向量搜尋，但取更多候選
    stage1_results = n_results if not enable_reranking else rerank_top_k

    search_results = _perform_hybrid_search(
        qdrant_client=qdrant_client,
        embedding_model=embeddings_manager,
        query=query,
        query_embedding=query_embedding,
        search_collections=search_collections,
        n_results=stage1_results,  # 取更多候選給 reranker
        search_mode=search_mode,
        metadata_extractor=metadata_extractor,
    )

    # Stage 2: Reranking
    if enable_reranking and search_results:
        from services.reranker_service import RerankerService

        reranker = get_reranker_instance()  # Singleton pattern

        reranked_results = reranker.rerank_results(
            query=query,
            results=search_results,
            top_k=n_results,
            instruction=rerank_instruction
        )

        # 轉換回原有格式
        search_results = [
            {
                **result.metadata,
                "content": result.content,
                "file_path": result.file_path,
                "score": result.rerank_score,
                "original_score": result.original_score,
                "reranked": True,
            }
            for result in reranked_results
        ]
```

### Phase 3: 環境配置

#### 3.3 更新 `.env` 設定

```bash
# Reranker Configuration
RERANKER_ENABLED=true
RERANKER_PROVIDER=transformers    # transformers | ollama | mlx
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
RERANKER_MAX_LENGTH=512
RERANKER_BATCH_SIZE=8
RERANK_TOP_K=50                   # Stage 1 候選數量

# Code-specific reranking instruction
RERANKER_CODE_INSTRUCTION="Given a code search query, determine if the code snippet is relevant for understanding or implementing the requested functionality."
```

### Phase 4: MCP Tool 更新

#### 3.4 更新 `src/tools/registry.py`

在 search tool 的 schema 中加入新參數：

```python
{
    "name": "search",
    "description": "Search indexed content with two-stage retrieval...",
    "inputSchema": {
        "type": "object",
        "properties": {
            # ... 現有參數 ...
            "enable_reranking": {
                "type": "boolean",
                "default": True,
                "description": "Enable cross-encoder reranking for improved accuracy"
            },
            "rerank_top_k": {
                "type": "integer",
                "default": 50,
                "minimum": 10,
                "maximum": 200,
                "description": "Number of candidates for Stage 1 before reranking"
            }
        }
    }
}
```

---

## 4. 實作時程

| Phase | 任務 | 預估時間 |
|-------|------|----------|
| 1 | RerankerService 核心實作 | 2-3 小時 |
| 2 | Search Pipeline 整合 | 1-2 小時 |
| 3 | 環境配置與測試 | 1 小時 |
| 4 | MCP Tool Schema 更新 | 30 分鐘 |
| 5 | 文件更新 | 30 分鐘 |

**總計：約 5-7 小時**

---

## 5. 效能考量

### 5.1 延遲預估

| 操作 | 預估延遲 |
|------|----------|
| Stage 1: Vector Search (50 candidates) | ~50ms |
| Stage 2: Reranking (50 candidates) | ~400ms (CPU) / ~100ms (MPS) |
| 總延遲 | ~450ms (CPU) / ~150ms (MPS) |

### 5.2 優化策略

1. **Batch Processing**: 批次處理 reranking 請求
2. **Lazy Loading**: 延遲載入 reranker model
3. **Caching**: 快取常見 query-document pair 的分數
4. **Optional Reranking**: 允許使用者關閉 reranking 以獲得更快回應

### 5.3 記憶體使用

- Qwen3-Reranker-0.6B: ~1.5GB VRAM/RAM
- 建議系統記憶體: 8GB+
- Apple Silicon MPS 加速: 顯著提升效能

---

## 6. 測試計劃

### 6.1 單元測試

```python
def test_reranker_basic():
    """Test basic reranking functionality."""
    reranker = RerankerService()

    query = "how to implement error handling"
    results = [
        {"content": "try: ... except Exception as e: ...", "score": 0.8},
        {"content": "print('hello world')", "score": 0.75},
    ]

    reranked = reranker.rerank_results(query, results)

    # Error handling code should rank higher after reranking
    assert reranked[0].content.startswith("try:")
```

### 6.2 整合測試

- 測試 search pipeline 完整流程
- 比較 reranking 前後的結果品質
- 測量延遲增加幅度

### 6.3 效能基準測試

- 測量 50/100/200 candidates 的 reranking 時間
- 比較 CPU vs MPS 效能
- 記憶體使用監控

---

## 7. 風險與緩解

| 風險 | 緩解策略 |
|------|----------|
| Reranker 延遲過高 | 可關閉 reranking、減少 top_k |
| 記憶體不足 | 使用較小模型 (0.6B) |
| Model 下載失敗 | 支援 Ollama fallback |
| 結果品質未提升 | A/B 測試驗證效果 |

---

## 8. 參考資料

- [Hands-on RAG with Qwen3 Embedding and Reranking Models using Milvus](https://milvus.io/blog/hands-on-rag-with-qwen3-embedding-and-reranking-models-using-milvus.md)
- [Qwen3 Embedding: Advancing Text Embedding and Reranking](https://qwenlm.github.io/blog/qwen3-embedding/)
- [Milvus Reranking Documentation](https://milvus.io/docs/reranking.md)
- [How to Run Qwen3 Embedding and Reranker Models Locally with Ollama](https://apidog.com/blog/qwen-3-embedding-reranker-ollama/)

---

## 9. 結論

引入兩階段 RAG 架構將顯著提升 Codebase RAG MCP Server 的搜尋品質，特別是對於：

1. **複雜程式碼查詢**: Cross-encoder 能更好理解 query 和 code 之間的語義關係
2. **多義詞消歧**: 透過深度上下文理解減少誤判
3. **Instruction-aware 搜尋**: 可根據不同任務類型優化結果

建議採用漸進式實作，先完成基礎功能，再逐步優化效能。
