# AGENTS.md

台灣防制洗錢法規問答（RAG）。以 Python、LangChain、LangGraph 實作，形式為 CLI。
設計文件在 `docs/specs/`，實作計畫在 `docs/plans/`。

## 目錄結構

```
aml_reg_rag/
├── pyproject.toml / uv.lock / README.md / .env.example / docker-compose.yml
├── data/corpus.toml       # 語料法規清單（pcode）
├── eval/
│   ├── curated.jsonl      # 人工題
│   └── synthetic.jsonl    # 自動生成題（reg eval generate 產生）
├── src/aml_reg_rag/
│   ├── cli.py / render.py # Typer 指令與 rich 顯示
│   ├── config.py          # Settings（pydantic-settings）
│   ├── models.py          # 共用資料模型
│   ├── llm.py             # 建立 structured output 模型、token 統計
│   ├── citations.py       # 引用驗證
│   ├── trace.py           # 問答 trace
│   ├── ingest/            # 下載與解析、依項款目切段、匯入流程
│   ├── retrieval/         # embedding、PGVectorStore 索引與 index_meta
│   ├── graph/             # LangGraph：state、nodes、prompts、組裝
│   └── evaluation/        # 題庫、自動生成、judge、指標、runner
├── tests/
└── var/                   # raw/、traces/、eval/（不進版控）
```

## 工具鏈

- **Python**：3.12 以上
- **套件管理**：`uv`（`uv add`、`uv run`），不直接使用 `pip install`
- **資料庫**：Postgres + pgvector，`docker compose up -d` 啟動，或以 `DATABASE_URL` 連線本機資料庫
- **Lint / Format**：`ruff`，line-length 100；`graph/prompts.py`、`evaluation/prompts.py` 豁免 E501，prompt 文字保持原樣
- **型別檢查**：`mypy src`，公開函式必須有型別標註
- **測試**：`pytest`；單元與流程測試使用 fake model 與 `FakeIndex`，不連資料庫與外部 API；串接真實資料庫、embedding 或 LLM 的測試標記 `@pytest.mark.integration`，預設不執行
- **設定**：`pydantic-settings` 讀取 `.env`，程式中不得寫死任何 key

```bash
uv sync
uv run pytest
uv run pytest -m integration   # 需要 DATABASE_URL；LLM 測試另需 API key 與已建立的索引
uv run ruff format src tests && uv run ruff check src tests && uv run mypy src
```

## 程式碼規範

- Clean code：單一職責、命名清楚、函式短小、避免過度抽象
- 資料模型使用 `pydantic.BaseModel`，graph state 使用 `TypedDict`
- 依賴方向單一：`cli` → `evaluation` → `graph` → `retrieval` / `citations` / `llm` → `models` / `config`；`ingest` → `retrieval`
- 不留死碼、未使用的 import、註解掉的程式碼

### 註解規則

- 只寫必要的註解，說明「為什麼」而非重述程式碼
- 每段註解不超過兩行
- docstring 同樣精簡，只在公開 API 或行為不直觀時撰寫

## LangChain / LangGraph 慣例

- State、Node、組裝分開放置，`build_graph()` 回傳 compiled graph，index 與 drafter 由外部注入
- 引用是否屬實由 `citations.verify` 以程式判斷，不交給 LLM；驗證失敗不自動刪改引用
- Prompt 集中於 `graph/prompts.py` 與 `evaluation/prompts.py`
- 模型透過 `llm.structured_model` 以 `init_chat_model("<provider>:<model>")` 建立，不寫死在 node 中
- v1 為單次問答，不使用 checkpointer

## LLM 與 embedding 設定

- `LLM_MODEL` 預設 `google_genai:gemini-3.8-flash`；`LLM_LIGHT_MODEL` 用於產生自動生成題
- `EVAL_JUDGE_MODEL` 獨立設定，不跟隨 `LLM_MODEL`，比較生成模型時量尺不變
- 不設定 temperature：Gemini 3 官方建議維持預設 1.0
- `EMBEDDING_MODEL` 以 `init_embeddings` 建立，維度參數依 provider 對應（`retrieval/embeddings.py`）；每組模型與維度各一張索引表
- API key 由各 provider SDK 從環境變數讀取，CLI 進入點以 `load_dotenv()` 載入 `.env`
- 成本估算表在 `evaluation/metrics.py` 的 `PRICING`，價格變動時一併更新

## 資料

- 語料來自全國法規資料庫 Open API，下載快取於 `var/raw/`；法規清單在 `data/corpus.toml`，以 pcode 為準
- `eval/curated.jsonl` 的預期條文標到「條」（`<pcode>:<條號>`），切段參數改變時不需修改
- `eval/synthetic.jsonl` 進版控，重新產生需 `--overwrite`

## 安全

- `.env` 不進版控，只提交 `.env.example`
- 不在 log、trace 或錯誤訊息中出現 API key；外部 API 錯誤只顯示例外類型與狀態碼
- 問題會送到 LLM 與 embedding API，不輸入客戶個資或案件機敏資訊
- docker compose 的資料庫只綁定 `127.0.0.1:55432`；避開 543x，因為本機的 Cloud SQL proxy 使用這些 port
- `reg ingest` 會刪表重建，`DATABASE_URL` 不可指向共用、測試或正式環境的資料庫
