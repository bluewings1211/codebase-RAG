# RAG 測試框架研究與 CI/CD 建議

> 本文件為 Codebase-RAG-MCP 專案的測試策略與 CI/CD 配置指南。

## 目錄

1. [RAG 測試框架總覽](#1-rag-測試框架總覽)
2. [RAGAS 框架詳細分析](#2-ragas-框架詳細分析)
3. [GitHub Actions CI 配置](#3-github-actions-ci-配置)
4. [Pytest Markers 配置](#4-pytest-markers-配置)
5. [RAG 品質測試模板](#5-rag-品質測試模板)
6. [測試執行指南](#6-測試執行指南)

---

## 1. RAG 測試框架總覽

### 1.1 主流框架比較

| 框架 | 最佳用途 | 關鍵優勢 | Ollama 支援 | 離線評估 |
|------|----------|----------|-------------|----------|
| **RAGAS** | Reference-free 評估 | 最新研究支持，核心 metrics 完整 | ✅ 支援 | ✅ 完全支援 |
| **DeepEval** | CI/CD 與單元測試 | pytest 整合，自我解釋 metrics | ✅ 支援 | ✅ 支援 |
| **TruLens** | 質性分析與監控 | Feedback functions, guardrails | ⚠️ 部分 | ⚠️ 有限 |
| **RAGChecker** | 細粒度診斷 | Claim-level entailment checks | ✅ 支援 | ✅ 支援 |
| **Evidently** | 全生命週期評估 | Web UI, 內建 metrics | ⚠️ 部分 | ✅ 支援 |

### 1.2 核心評估指標（Core Four Metrics）

根據 2025 年最佳實踐，以下四個指標涵蓋 95% 以上的實際評估需求：

| 指標 | 說明 | 評估對象 |
|------|------|----------|
| **Context Precision** | 檢索結果的排序品質（相關內容是否排在前面） | Retriever |
| **Context Recall** | 檢索內容的完整性（是否包含所有必要資訊） | Retriever |
| **Faithfulness** | 生成結果是否基於檢索到的內容（幻覺偵測） | Generator |
| **Answer Relevancy** | 回答與問題的相關性 | End-to-End |

---

## 2. RAGAS 框架詳細分析

### 2.1 系統需求

```
Python 版本: ≥ 3.9
最新版本: ragas-0.4.0 (2025-12-03)
```

### 2.2 依賴套件

RAGAS 及測試相關依賴已加入 `pyproject.toml` 的 `dev` dependency group：

```bash
# 安裝開發環境（包含 RAGAS 和測試工具）
uv sync --dev

# 僅安裝生產環境依賴
uv sync
```

**dev 群組包含的測試依賴：**
| 套件 | 用途 |
|------|------|
| `pytest` | 測試框架 |
| `pytest-cov` | 覆蓋率報告 |
| `pytest-xdist` | 平行測試執行 |
| `ragas` | RAG 評估框架 |
| `datasets` | RAGAS 依賴 |
| `ruff`, `black` | Linting & Formatting |
| `pre-commit` | Git hooks |

### 2.3 支援的 LLM 模型

#### 直接支援的提供者

| 提供者 | 支援狀態 | 配置方式 |
|--------|----------|----------|
| OpenAI | ✅ 完整支援 | 內建 factory |
| Anthropic | ✅ 完整支援 | 內建 factory |
| Google Gemini | ✅ 完整支援 | 內建 factory |
| **Ollama** | ✅ 完整支援 | LiteLLM adapter |
| HuggingFace | ✅ 支援 | LiteLLM adapter |
| vLLM | ✅ 支援 | LiteLLM adapter |

#### 本地模型建議

| 用途 | 建議模型 | 最低需求 |
|------|----------|----------|
| 評估 LLM | `llama3.1:8b`, `mistral:7b` | 7B+ 參數 |
| Embedding | `nomic-embed-text` | - |
| 資源受限環境 | `llama3.2:3b` | 3B+ 參數 |

### 2.4 Ollama 配置範例

```python
from ragas.llms import llm_factory
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# 配置 Ollama 作為 LLM 提供者
llm = llm_factory(
    "llama3.1:8b",  # Ollama 中的模型名稱
    provider="ollama",
    base_url="http://localhost:11434"
)

# 執行評估
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=llm
)
```

### 2.5 離線評估支援

**完全支援離線評估**，但需注意：

| 面向 | 說明 |
|------|------|
| 運算需求 | 所有 metrics 需要 LLM 推理（本地執行） |
| 處理時間 | 取決於本地硬體（建議使用 GPU） |
| 品質考量 | 小於 7B 參數的模型可能產生不穩定的評分 |
| 資料隱私 | 完全本地處理，適合敏感資料 |

### 2.6 已知限制與解決方案

| 問題 | 解決方案 |
|------|----------|
| 預設初始化 OpenAI | 使用 `llm_factory` 明確指定 Ollama |
| 評分不穩定 | 使用 7B+ 參數模型 |
| 自我評估偏差 | 使用不同模型進行生成與評估 |
| Embedding 不一致 | 確保 indexing 與評估使用相同的 embedding 模型 |

---

## 3. GitHub Actions CI 配置

### 3.1 完整配置檔

建立 `.github/workflows/ci.yml`：

```yaml
name: CI Pipeline

on:
  pull_request:
    branches: [main, dev]
  push:
    branches: [main]
  schedule:
    # 每週一 UTC 00:00 執行完整測試
    - cron: '0 0 * * 1'

env:
  PYTHONPATH: src
  UV_SYSTEM_PYTHON: 1

jobs:
  # ============================================
  # Stage 1: Quick Checks (< 2 分鐘)
  # ============================================
  lint:
    name: Lint & Format Check
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync --dev

      - name: Run Ruff linter
        run: uv run ruff check src/

      - name: Run Black format check
        run: uv run black --check src/

  # ============================================
  # Stage 2: Fast Unit Tests (< 5 分鐘)
  # ============================================
  test-unit:
    name: Unit Tests
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync --dev

      - name: Run fast unit tests
        run: |
          uv run pytest src/tests/ \
            -m "not slow and not integration and not rag_quality and not requires_qdrant and not requires_ollama" \
            --tb=short \
            -q \
            --ignore=src/tests/test_end_to_end_workflow.py \
            --ignore=src/tests/test_performance_benchmarks.py

  # ============================================
  # Stage 3: Core Feature Tests (< 10 分鐘)
  # ============================================
  test-core:
    name: Core Feature Tests
    runs-on: ubuntu-latest
    needs: test-unit
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync --dev

      - name: Run core feature tests
        run: |
          uv run pytest \
            src/tests/test_code_parser_service.py \
            src/tests/test_intelligent_chunking.py \
            src/tests/test_syntax_error_handling.py \
            src/tests/test_file_discovery_service.py \
            src/tests/test_search_bug_fixes.py \
            -v \
            --tb=short

  # ============================================
  # Stage 4: Integration Tests (僅在 merge 到 main 時執行)
  # ============================================
  test-integration:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: test-core
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
          - 6334:6334
        options: >-
          --health-cmd "curl -f http://localhost:6333/health || exit 1"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    env:
      QDRANT_HOST: localhost
      QDRANT_PORT: 6333
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync --dev

      - name: Wait for Qdrant to be ready
        run: |
          for i in {1..30}; do
            if curl -s http://localhost:6333/health > /dev/null; then
              echo "Qdrant is ready"
              break
            fi
            echo "Waiting for Qdrant... ($i/30)"
            sleep 2
          done

      - name: Run integration tests
        run: |
          uv run pytest \
            src/tests/test_indexing_pipeline.py \
            -v \
            -m "integration or requires_qdrant" \
            --tb=short

  # ============================================
  # RAG Quality Tests (手動觸發或排程執行)
  # ============================================
  test-rag-quality:
    name: RAG Quality Evaluation
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
    env:
      QDRANT_HOST: localhost
      QDRANT_PORT: 6333
      # 禁用需要 Ollama 的測試（CI 環境無 Ollama）
      SKIP_OLLAMA_TESTS: "true"
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync --dev

      - name: Run RAG quality tests (basic metrics only)
        run: |
          uv run pytest \
            src/tests/test_rag_quality.py \
            -v \
            -m "rag_quality and not requires_ollama" \
            --tb=short
        continue-on-error: true

      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: rag-quality-results
          path: |
            rag_quality_report.json
            pytest_report.xml
          retention-days: 30

  # ============================================
  # Performance Benchmarks (僅排程執行)
  # ============================================
  test-performance:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
    env:
      QDRANT_HOST: localhost
      QDRANT_PORT: 6333
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync --dev

      - name: Run performance benchmarks
        run: |
          uv run pytest \
            src/tests/test_performance_benchmarks.py \
            -v \
            --tb=short
        continue-on-error: true
```

### 3.2 Pipeline 流程圖

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GitHub Actions CI Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PR/Push Event                                                       │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────┐                                                     │
│  │   Stage 1   │ Lint & Format Check                                │
│  │   (< 2m)    │ ├─ ruff check                                      │
│  └──────┬──────┘ └─ black --check                                   │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────┐                                                     │
│  │   Stage 2   │ Unit Tests (mocked, fast)                          │
│  │   (< 5m)    │ └─ pytest -m "not slow and not integration"        │
│  └──────┬──────┘                                                     │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────┐                                                     │
│  │   Stage 3   │ Core Feature Tests                                 │
│  │   (< 10m)   │ ├─ code_parser_service                             │
│  └──────┬──────┘ ├─ intelligent_chunking                            │
│         │        └─ search_bug_fixes                                 │
│         │                                                            │
│         ▼ (only on merge to main)                                   │
│  ┌─────────────┐                                                     │
│  │   Stage 4   │ Integration Tests (with Qdrant)                    │
│  │   (< 20m)   │ └─ indexing_pipeline                               │
│  └─────────────┘                                                     │
│                                                                      │
│  Scheduled/Manual Only:                                              │
│  ┌─────────────┐  ┌─────────────┐                                   │
│  │ RAG Quality │  │ Performance │                                   │
│  │   Tests     │  │ Benchmarks  │                                   │
│  └─────────────┘  └─────────────┘                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Pytest Markers 配置

### 4.1 pyproject.toml 配置

將以下配置新增至 `pyproject.toml`：

```toml
[tool.pytest.ini_options]
# 測試路徑
testpaths = ["src/tests"]

# 預設選項
addopts = "-v --tb=short -ra"

# Marker 定義
markers = [
    "slow: 標記為慢速測試 (使用 '-m \"not slow\"' 排除)",
    "integration: 需要外部服務的整合測試 (Qdrant, Ollama)",
    "rag_quality: RAG 品質評估測試",
    "requires_qdrant: 需要 Qdrant 資料庫的測試",
    "requires_ollama: 需要 Ollama 服務的測試",
    "requires_reranker: 需要 reranker 模型的測試",
    "unit: 快速單元測試 (無外部依賴)",
    "smoke: 煙霧測試 (基本功能驗證)",
]

# 過濾警告
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
]

# 最低覆蓋率要求 (需安裝 pytest-cov)
# [tool.coverage.run]
# source = ["src"]
# omit = ["src/tests/*", "src/**/test_*.py"]
```

### 4.2 為現有測試檔案加上 Markers

#### 範例：test_code_parser_service.py

```python
import pytest

# 在檔案開頭或 class 上方加上 marker
@pytest.mark.unit
class TestCodeParserService:
    """Code parser service unit tests."""

    def test_parse_python_file(self):
        # ...
        pass

    @pytest.mark.slow
    def test_parse_large_file(self):
        # 這個測試較慢，標記為 slow
        pass
```

#### 範例：test_indexing_pipeline.py

```python
import pytest

@pytest.mark.integration
@pytest.mark.requires_qdrant
class TestIndexingPipeline:
    """Integration tests requiring Qdrant."""

    def test_full_indexing_workflow(self):
        # ...
        pass
```

#### 範例：test_embedding_service.py

```python
import pytest

class TestEmbeddingService:
    """Embedding service tests."""

    @pytest.mark.unit
    def test_generate_embeddings_mocked(self):
        """Mocked test - no external service needed."""
        pass

    @pytest.mark.requires_ollama
    def test_generate_embeddings_real(self):
        """Real test - requires Ollama running."""
        pass
```

### 4.3 Marker 使用建議

| Marker | 適用情境 | CI 階段 |
|--------|----------|---------|
| `@pytest.mark.unit` | 快速、無外部依賴的測試 | Stage 2 |
| `@pytest.mark.slow` | 執行時間 > 10 秒的測試 | 排除於 PR |
| `@pytest.mark.integration` | 需要外部服務的測試 | Stage 4 |
| `@pytest.mark.requires_qdrant` | 需要 Qdrant 的測試 | Stage 4 |
| `@pytest.mark.requires_ollama` | 需要 Ollama 的測試 | 本地/排程 |
| `@pytest.mark.rag_quality` | RAG 品質評估測試 | 排程執行 |
| `@pytest.mark.smoke` | 基本功能煙霧測試 | 所有階段 |

---

## 5. RAG 品質測試模板

詳見 `src/tests/test_rag_quality.py`（獨立檔案）。

### 5.1 主要功能

- **Golden Dataset**: 預定義的測試案例集
- **基本 Metrics**: 不需 LLM 的基本評估指標
- **進階 Metrics**: 使用 RAGAS/Ollama 的完整評估
- **報告生成**: JSON 格式的測試報告

### 5.2 測試類別

| 類別 | 說明 | 依賴 |
|------|------|------|
| `TestBasicSearchQuality` | 基本搜尋品質測試 | Qdrant |
| `TestRetrievalMetrics` | 檢索指標測試 | Qdrant |
| `TestRAGASIntegration` | RAGAS 整合測試 | Qdrant + Ollama |
| `TestSearchRobustness` | 搜尋穩健性測試 | Qdrant |

---

## 6. 測試執行指南

### 6.1 常用指令

```bash
# 執行所有快速測試 (排除慢速和整合測試)
uv run pytest src/tests/ -m "not slow and not integration"

# 只執行單元測試
uv run pytest src/tests/ -m "unit"

# 執行需要 Qdrant 的測試
uv run pytest src/tests/ -m "requires_qdrant"

# 執行 RAG 品質測試 (不需要 Ollama 的部分)
uv run pytest src/tests/test_rag_quality.py -m "rag_quality and not requires_ollama"

# 執行完整 RAG 品質測試 (需要 Ollama)
uv run pytest src/tests/test_rag_quality.py -m "rag_quality"

# 執行特定測試檔案
uv run pytest src/tests/test_code_parser_service.py -v

# 執行煙霧測試
uv run pytest src/tests/ -m "smoke" -v

# 產生覆蓋率報告
uv run pytest src/tests/ --cov=src --cov-report=html

# 執行測試並產生 JUnit XML 報告
uv run pytest src/tests/ --junitxml=pytest_report.xml
```

### 6.2 本地開發流程

```bash
# 1. 開發前：執行快速測試確認基礎功能
uv run pytest src/tests/ -m "unit" -q

# 2. 開發中：執行相關測試
uv run pytest src/tests/test_search_tools.py -v

# 3. 提交前：執行完整快速測試套件
uv run pytest src/tests/ -m "not slow and not integration" --tb=short

# 4. 重大變更：執行整合測試 (需要 Qdrant)
docker run -d -p 6333:6333 qdrant/qdrant
uv run pytest src/tests/ -m "integration" -v
```

### 6.3 CI/CD 觸發條件

| 事件 | 執行的測試 |
|------|------------|
| PR 開啟/更新 | Stage 1-3 (lint, unit, core) |
| Merge 到 main | Stage 1-4 (包含 integration) |
| 每週排程 | 全部測試 (包含 RAG quality, performance) |
| 手動觸發 | 可選擇性執行任何測試套件 |

---

## 附錄

### A. 相關文件

- [RAGAS 官方文件](https://docs.ragas.io/)
- [DeepEval 文件](https://docs.confident-ai.com/)
- [pytest 文件](https://docs.pytest.org/)
- [GitHub Actions 文件](https://docs.github.com/en/actions)

### B. 故障排除

#### Ollama 連線問題

```bash
# 確認 Ollama 正在執行
ollama list

# 啟動 Ollama 服務
ollama serve

# 測試連線
curl http://localhost:11434/api/tags
```

#### Qdrant 連線問題

```bash
# 使用 Docker 啟動 Qdrant
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 測試連線
curl http://localhost:6333/health
```

### C. 版本歷史

| 版本 | 日期 | 變更 |
|------|------|------|
| 1.0 | 2025-12-11 | 初始版本 |
