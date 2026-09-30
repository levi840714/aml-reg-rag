# AML 法規問答設計文件：虛擬資產防制洗錢法規 RAG

- 日期：2026-09-30
- 狀態：v1 設計完成，待實作

## 1. 目標

輸入自然語言問題，從台灣防制洗錢相關法規中檢索條文，產出每個論點都附條號與原文引述的回答；引用由程式驗證，條文不足以回答時拒答。

重點：依法規結構切段、向量檢索、可驗證的引用、檢索與生成分開量測的 eval。v1 建立 baseline，後續版本的每項改進都以此對照。

### 範圍

- 包含：6 部現行法規中文版（全國法規資料庫 Open API）、向量檢索（Gemini embedding + pgvector）、結構化生成與引用驗證、CLI、人工題與自動生成題兩套 eval
- 不包含：BM25 / hybrid、reranker、query rewriting、法規簡稱與舊名對應、英文版、自律規範 PDF、歷史版本與生效日、條文互引展開、多輪對話、agentic 重新檢索、前端（見第 9 節）

### 語料

| 法規 | pcode | 層級 | 條數 |
| ---- | ----- | ---- | ---- |
| 洗錢防制法 | G0380131 | 法律 | 31 |
| 資恐防制法 | I0030047 | 法律 | 16 |
| 提供虛擬資產服務之事業或人員防制洗錢及打擊資恐辦法 | G0380268 | 命令 | 18 |
| 提供虛擬資產服務之事業或人員洗錢防制登記辦法 | G0400162 | 命令 | 31 |
| 金融機構及提供虛擬資產服務之事業或人員防制詐欺犯罪危害應遵循事項辦法 | G0380271 | 命令 | 73 |
| 金融機構防制洗錢辦法 | G0380252 | 命令 | 16 |

共 185 條、約 5.4 萬字。金融機構防制洗錢辦法與 VASP 辦法內容相近，作為檢索的干擾項。法規條文與機關譯本依著作權法第 9 條不受著作權保護。

### 安全原則

- 回答只依據檢索到的條文，引用是否屬實由程式驗證，不交給 LLM
- 問題會送到 LLM 與 embedding API，README 提醒勿輸入客戶個資或案件機敏資訊
- 不在 log、trace 或錯誤訊息中出現 API key
- docker compose 的資料庫只綁定 `127.0.0.1`
- 輸出僅供查詢法規依據，不構成法律意見（README 註明）

## 2. 架構

**匯入**（`reg ingest`）：

```
fetch ──→ parse ──→ chunk ──→ embed + store
```

**問答**（`reg ask`），以 LangGraph 實作：

```
retrieve ──→ generate ──→ verify ──(通過)─────────────→ END
                ↑            │
                └──(未通過，重試 1 次，附問題清單)
                             └──(重試後仍未通過)──→ END（標記驗證失敗）
```

LangChain 元件分工：

| 環節 | 實作 |
| ---- | ---- |
| 載入、切段 | 自行實作，輸出 `langchain_core.documents.Document`（全國法規資料庫 JSON 無現成 loader，內建 splitter 不理解項款目層級） |
| Embedding | `init_embeddings("<provider>:<model>")` |
| 向量庫 | `langchain-postgres` 的 `PGVectorStore`，自訂 metadata 欄位 |
| Prompt | `ChatPromptTemplate`，集中於 `graph/prompts.py` |
| 生成 | `init_chat_model` + `with_structured_output` |
| 流程 | LangGraph `StateGraph` |
| 引用驗證 | 純 Python |

v1 為單次問答，無中斷恢復需求，不使用 checkpointer。

## 3. 專案結構

```
aml_reg_rag/
├── AGENTS.md / CLAUDE.md / README.md / .gitignore / .env.example
├── pyproject.toml / uv.lock
├── docker-compose.yml         # pgvector/pgvector:pg18，綁定 127.0.0.1:5433
├── data/corpus.toml           # 語料法規清單
├── eval/
│   ├── curated.jsonl          # 人工題
│   └── synthetic.jsonl        # 自動生成題（進版控）
├── docs/specs/ / docs/plans/
├── src/aml_reg_rag/
│   ├── __main__.py / cli.py
│   ├── render.py              # rich 呈現：檢索結果、回答、eval 摘要
│   ├── config.py
│   ├── models.py              # LawArticle、Chunk、RetrievedChunk、Citation、Point、Answer
│   ├── llm.py                 # 建立 structured output 模型、token 統計
│   ├── citations.py           # 引用驗證
│   ├── trace.py
│   ├── ingest/
│   │   ├── source.py          # 下載與解析全國法規資料庫 JSON
│   │   ├── chunker.py         # 依項款目結構切段
│   │   └── pipeline.py        # 匯入流程
│   ├── retrieval/
│   │   ├── embeddings.py      # 建立 embedding 模型、維度參數對應
│   │   └── index.py           # ChunkIndex Protocol、PGVectorStore 實作、index_meta
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── prompts.py
│   │   └── build.py
│   └── evaluation/
│       ├── dataset.py         # 讀取與驗證題庫
│       ├── prompts.py         # 自動生成題與 judge 的 prompt
│       ├── generate.py        # 自動生成題
│       ├── judge.py           # 要點涵蓋判斷
│       ├── metrics.py         # 檢索與生成指標、PRICING
│       └── runner.py
├── tests/
└── var/                       # raw/、traces/、eval/（不進版控）
```

相依方向：`cli` → `evaluation` → `graph` → `retrieval` / `citations` / `llm` → `models` / `config`；`ingest` → `retrieval`。只有 `graph` 與 `evaluation` 經由 `llm` 接觸 LLM。

## 4. 元件

### 4.1 config

`pydantic-settings` 讀取 `.env`：

| 設定 | 預設值 | 說明 |
| ---- | ------ | ---- |
| `LLM_MODEL` | `google_genai:gemini-3.8-flash` | 回答生成 |
| `LLM_LIGHT_MODEL` | `google_genai:gemini-3.5-flash-lite` | 產生自動生成題 |
| `EVAL_JUDGE_MODEL` | `google_genai:gemini-3.8-flash` | 要點涵蓋判斷；固定預設，不跟隨 `LLM_MODEL`，比較生成模型時量尺不變 |
| `EMBEDDING_MODEL` | `google_genai:gemini-embedding-001` | 傳給 `init_embeddings` |
| `EMBEDDING_DIM` | 1536 | 需 ≤ 2000（pgvector HNSW 上限） |
| `DATABASE_URL` | `postgresql+psycopg://reg:reg@localhost:5433/aml_reg` | 本機 Postgres 或 docker compose |
| `TOP_K` | 6 | 檢索段數 |
| `CHUNK_MAX_CHARS` | 600 | 切段上限（內文字元數，含前綴、不含標頭） |
| `LLM_TIMEOUT` | 120 | 單次 LLM 呼叫逾時（秒） |

不設定 temperature：Gemini 3 官方建議維持預設 1.0，調低可能造成重複輸出或表現變差。`thinking_level` 使用模型預設值。

### 4.2 ingest

**source**

- 下載 `https://law.moj.gov.tw/api/Ch/Law/json` 與 `https://law.moj.gov.tw/api/Ch/Order/json`（zip，內含 UTF-8 BOM 的 JSON），快取於 `var/raw/`；`--refresh` 才重新下載
- 依 `data/corpus.toml` 篩選法規：

  ```toml
  [[laws]]
  pcode = "G0380131"
  name = "洗錢防制法"
  ```

  `name` 僅供閱讀；與來源名稱不同時顯示警告（法規可能已更名），以 pcode 為準
- 取 `ArticleType == "A"` 的條文，並以 `"C"` 記錄目前所屬章名；條號 `第 5-1 條` 正規化為 `5-1`
- 記錄來源的 `UpdateDate` 與各法規的 `LawModifiedDate`、`LawLevel`

**chunker**

條文依行解析為層級樹：

| 層級 | 判斷（行首） |
| ---- | ---- |
| 款 | `^[一二三四五六七八九十百]+、` |
| 目 | `^[（(][一二三四五六七八九十百]+[）)]` |
| 子目 | `^\d+[.．]` |
| 項 | 不符合以上者，開始新的一項 |

切段規則，依序套用（長度以內文計算，含前綴、不含標頭）：

1. 整條 ≤ `CHUNK_MAX_CHARS` → 一段
2. 否則以項為單位，相鄰項依序合併，合併後不超過上限才併入同一段
3. 單一項仍超過 → 以該項第一行（引言）為前綴，以款為單位依序合併
4. 單一款仍超過 → 以「項引言 + 款第一行」為前綴，以目為單位依序合併；子目隨所屬目，不再切分
5. 無法再依結構切分者保留為一段，ingest 摘要列出超長段數

每段 `page_content` 第一行為標頭，第二行起為內文（含前綴）：

```
【提供虛擬資產服務之事業或人員防制洗錢及打擊資恐辦法｜第 9 條 第 1 項】
<內文>
```

有章名時標頭加入章名。標頭與內文一起 embedding，法規名稱是區分金融機構版與 VASP 版的主要訊號。

**chunk_id 與 locator**

| 範圍 | chunk_id | locator |
| ---- | -------- | ------- |
| 整條 | `G0380268:9` | （空） |
| 項 | `G0380268:9:p1-2` | 第 1 項至第 2 項 |
| 款 | `G0380252:3:p1:k2-4` | 第 1 項第 2 款至第 4 款 |
| 目 | `G0380252:3:p1:k2:m1-3` | 第 1 項第 2 款第 1 目至第 3 目 |

單一單位時不寫範圍（`p1`、第 1 項）。編號為在上層中的順序（從 1 起算）。

**pipeline**

1. 刪除該索引表的 `index_meta` 紀錄
2. 重建索引表並寫入所有段落（每次全量重建）
3. 建立 HNSW 索引（cosine）
4. 寫入 `index_meta`

中途失敗時 `index_meta` 不存在，`search` / `ask` 會要求重新 ingest。第一批 embedding 回傳後檢查向量長度等於 `EMBEDDING_DIM`。

### 4.3 retrieval

**embeddings**

- `init_embeddings(EMBEDDING_MODEL, **dim_kwargs)`；維度參數依 provider 對應：`google_genai` → `output_dimensionality`、`openai` → `dimensions`，其他 provider 啟動時報錯
- `gemini-embedding-001` 查詢用 `RETRIEVAL_QUERY`、文件用 `RETRIEVAL_DOCUMENT`（確認 LangChain 預設行為，未自動帶入則明確指定）
- 使用 cosine 距離，降維後的向量不需另外正規化

**index**

- 索引表名稱：`chunks_<provider>_<model>_d<dim>`，非英數字元轉為 `_`，例如 `chunks_google_genai_gemini_embedding_001_d1536`；超過 63 字元時截斷並加 8 碼 hash
- `PGVectorStore` 以 `chunk_id` 為 id 欄位，metadata 欄位：

| 欄位 | 型別 | 說明 |
| ---- | ---- | ---- |
| `pcode` | TEXT | 法規代碼 |
| `law_name` | TEXT | 法規名稱 |
| `law_level` | TEXT | 法律 / 命令 |
| `article_no` | TEXT | 條號，如 `5-1` |
| `chapter` | TEXT | 章名，可為空 |
| `locator` | TEXT | 條內位置 |
| `law_modified_date` | DATE | 法規最後修正日 |
| `source_url` | TEXT | `https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode=<pcode>&flno=<article_no>` |

- `index_meta` 表以 SQL 直接維護：`table_name`（PK）、`embedding_model`、`embedding_dim`、`chunk_max_chars`、`source_update_date`、`chunk_count`、`ingested_at`
- `ChunkIndex` Protocol：`search(query, k) -> list[RetrievedChunk]`；`PGVectorStore` 實作以 `similarity_search_with_score` 取回，分數轉為 cosine 相似度（1 − 距離）
- `RetrievedChunk`：`chunk_id`、`article_key`（`<pcode>:<article_no>`）、`law_name`、`article_no`、`locator`、`source_url`、`body`（不含標頭）、`score`
- 開啟索引時比對 `index_meta` 與目前的 `EMBEDDING_MODEL`、`EMBEDDING_DIM`，不存在或不一致時報錯；`CHUNK_MAX_CHARS` 不一致只警告（索引仍可用，但反映的是舊的切段設定）

### 4.4 graph

**AskState**

| 欄位 | 說明 |
| ---- | ---- |
| `question` | 使用者問題 |
| `k` | 檢索段數 |
| `chunks` | 檢索結果，依序標為 `S1`…`Sk` |
| `answer` | 最新的 `Answer`，解析失敗為 `None` |
| `problems` | 最近一次驗證的問題清單 |
| `attempts` | generate 執行次數 |
| `verified` | 是否通過驗證 |
| `input_tokens`、`output_tokens` | 累計 token，output 含 thinking token |

**節點**

| 節點 | 行為 |
| ---- | ---- |
| retrieve | `index.search(question, k)` |
| generate | 以 `S1`…`Sk` 標籤呈現段落，呼叫 structured output；重試時附上前次的問題清單。解析失敗時 `answer = None`，交由 verify 判定 |
| verify | 呼叫 `citations.verify`；通過則結束，未通過且 `attempts < 2` 回到 generate，否則以 `verified = False` 結束 |

`build_graph(index, drafter)` 回傳 compiled graph，依賴由外部注入；drafter 為 `with_structured_output(Answer, include_raw=True)` 的模型。

**輸出結構**

```python
class Citation(BaseModel):
    source: str        # "S2"
    quote: str         # 從該段落逐字引述

class Point(BaseModel):
    statement: str
    citations: list[Citation]

class Answer(BaseModel):
    answerable: bool
    points: list[Point]
    missing: str | None  # 拒答時說明條文缺少什麼
```

不另設總結欄位：每句都須可追溯到條文，總結無法由程式驗證。第一個論點直接回答問題。

**prompts**：只依據提供的段落作答；每個論點至少一個引用，quote 逐字複製且至少 8 字；段落不足以回答時 `answerable = false` 並說明缺少的規範；不推論條文未寫明的罰則、金額或期限；注意法規的適用對象（金融機構或虛擬資產服務商）；以繁體中文回答。段落以「`[S1]` 標頭 + 內文」呈現。

### 4.5 citations

`verify(answer, chunks) -> list[str]`，回傳問題清單，空清單代表通過：

- `answer` 為 `None` → 輸出格式錯誤
- `answerable = true`：`points` 非空；每個 point 至少一個 citation；`source` 必須是本次的 `S` 標籤；`quote` 經 NFKC 正規化並移除空白後，須為對應段落內文（同樣正規化）的子字串，且至少 8 字
- `answerable = false`：`points` 為空、`missing` 非空

驗證失敗時不自動刪除有問題的引用，原樣輸出並標記。

### 4.6 trace

每次 `ask` 追加一筆至 `var/traces/<YYYY-MM-DD>.jsonl`：時間、問題、索引表名稱、模型、檢索結果（`chunk_id`、分數）、`Answer`、`verified`、`problems`、`attempts`、token、耗時。記錄 top-1 分數，供 v2 評估是否加入分數門檻。

## 5. CLI

| 指令 | 說明 |
| ---- | ---- |
| `reg ingest [--refresh]` | 建立索引，完成後顯示法規數、條數、段數、超長段數 |
| `reg search "<問題>" [--k N]` | 只檢索不呼叫 LLM：排名、相似度、chunk_id、法規與條號、內文開頭 |
| `reg ask "<問題>" [--k N] [--show-context]` | 回答：論點附 [1]、[2] 與引述；最後列出來源（法規名稱、條號、locator、連結）。拒答時顯示缺少什麼並列出檢索到的條文；驗證失敗時顯示警告與問題清單 |
| `reg eval generate [--overwrite]` | 產生自動生成題 |
| `reg eval run [--retrieval-only] [--model <provider:model>] [--k N] [--repeat N]` | 執行 eval |

`[1]`、`[2]` 依來源首次出現的順序編號。

## 6. Eval

### 6.1 題庫

**人工題** `eval/curated.jsonl`：

```json
{"id": "c-007", "question": "…", "category": "distractor", "answerable": true,
 "expected_articles": ["G0380268:9"], "forbidden_articles": ["G0380252:3"],
 "key_points": ["…"], "source": "claude_draft", "reviewed": false, "note": "…"}
```

| 欄位 | 說明 |
| ---- | ---- |
| `category` | `lookup` / `scenario` / `cross_law` / `distractor` / `refusal` |
| `expected_articles` | 預期條文，標到條（`<pcode>:<article_no>`），切段參數改變時題庫不需修改；`refusal` 為空 |
| `forbidden_articles` | 引用即算錯法規，選填 |
| `key_points` | 答案應涵蓋的要點；`refusal` 為空 |
| `source` / `reviewed` | 起草者與是否經使用者審核 |

第一批約 20 題由 Claude 對照實際條文起草（`source: claude_draft`、`reviewed: false`），使用者之後審核與補充：

| 類別 | 題數 |
| ---- | ---- |
| `lookup` 條文查詢 | 5 |
| `scenario` 情境判斷 | 5 |
| `cross_law` 跨法規 | 3 |
| `distractor` 干擾辨識 | 4 |
| `refusal` 應拒答（課稅、自律規範內容、外國法規） | 3 |

**自動生成題** `eval/synthetic.jsonl`：`reg eval generate` 以 `LLM_LIGHT_MODEL` 每條條文產生 1 題，少於 30 字的條文（如施行日期）不出題。prompt 要求以實務人員口吻描述需求、不出現條號、不照抄條文句子。欄位：`id`、`question`、`expected_articles`、`generator_model`、`created_at`。最多 4 個並行呼叫。檔案進版控，已存在時需 `--overwrite` 才重新產生。

**驗證**：`eval run` 開始時檢查 id 不重複、類別合法、`refusal` 題 `answerable = false` 且無預期條文、所有 `expected_articles` 與 `forbidden_articles` 存在於索引中；不符即報錯並列出。

### 6.2 指標

**檢索**（人工題中非 `refusal` 的題目與全部自動生成題；只需 embedding）：

- 將 top-k 段落依序轉為條文排名（同條取首次出現的名次）
- Hit@1、Hit@3、Hit@k：前 n 名是否有任一預期條文
- Recall@k：預期條文被找到的比例（跨法規題有多條）
- MRR：第一個預期條文名次的倒數

**生成**（全部人工題，執行完整 graph）：

| 指標 | 計算 |
| ---- | ---- |
| 驗證通過率 | `verified` |
| 拒答判斷 | 正確 / 該答卻拒答 / 該拒卻答，分開統計 |
| 引用正確 | 是否引用任一預期條文；是否引用禁止條文 |
| 要點涵蓋率 | `EVAL_JUDGE_MODEL` 逐一判斷 key point 是否涵蓋並附理由；只計預期與實際皆為可回答的題目 |
| 成本與耗時 | LLM token（含 thinking）× `PRICING`，涵蓋生成與 judge；未知模型成本記為空值並警告。embedding 費用極低，不計入 |

**`--repeat N`**：生成階段每題執行 N 次（檢索為確定性，只跑一次），報告各指標平均值、每題的一致率（拒答判斷與驗證結果是否相同），並列出不穩定的題目。

**`--model`**：只覆寫生成模型；judge 固定使用 `EVAL_JUDGE_MODEL`。

### 6.3 輸出

- 依題庫與類別分組顯示；列出人工題中已審核與未審核的題數
- 結果寫入 `var/eval/<timestamp>-<model>.json`：設定快照（各模型、`TOP_K`、`CHUNK_MAX_CHARS`）、`index_meta`、每題明細
- 比較 embedding 模型、維度或切段大小時，修改設定、重新 ingest 後再執行

### 6.4 v1 完成標準

流程可端到端執行、eval 可重複執行，並將第一次結果作為 baseline 寫入 README。v1 不設及格分數。

## 7. 錯誤處理

| 情境 | 處理 |
| ---- | ---- |
| 全國法規資料庫下載失敗 | 有快取則使用並顯示快取日期；無快取則報錯 |
| `corpus.toml` 的 pcode 在來源中不存在 | 報錯並列出 |
| 條文內容為空 | 跳過並警告 |
| 資料庫連線失敗 | 提示檢查 `DATABASE_URL` 或執行 `docker compose up -d` |
| 缺少 pgvector 擴充 | 提示安裝擴充或改用 docker compose |
| 索引不存在或與目前 embedding 設定不一致 | 提示重新 `reg ingest` |
| embedding 或 LLM API 錯誤 | 模型的 `max_retries` 重試，仍失敗則報錯；只顯示例外類型與狀態碼 |
| structured output 解析失敗 | 視為驗證未通過，走同一個重試 |
| eval 單題失敗 | 記錄錯誤並繼續，報告列出錯誤題數 |

## 8. 測試

- 單元測試（無網路、無資料庫）：
  - source：以真實資料擷取的小型 fixture，含金融機構防制洗錢辦法第 3 條（2,285 字、四層結構）與第 5-1 條
  - chunker：各切段規則、前綴、chunk_id 與 locator
  - citations：各驗證規則、NFKC 正規化、8 字下限
  - retrieval：維度參數對應、索引表命名
  - evaluation：各指標計算、題庫驗證
- 流程測試（fake `ChunkIndex` + fake chat model）：直接通過、重試後通過、重試後仍失敗、拒答
- `@pytest.mark.integration`（預設不執行）：真實 Postgres 的 ingest 與 search；真實 embedding 與 LLM 回答一題

## 9. 後續版本

| 版本 | 內容 |
| ---- | ---- |
| v2 | 檢索改進：BM25 + 向量（`EnsembleRetriever`）、reranker、query rewriting、法規簡稱與舊名對應、英文版與自律規範 PDF；比較 `gemini-embedding-2`、`thinking_level`、分數門檻 |
| v3 | agentic 重新檢索、條文互引展開、多輪對話（checkpointer）、前端（FastAPI + `frontend/`：對話介面、引用與原文對照、使用者回饋回流為 eval 題目） |
| v4 | 與 kyt_agent 整合：報告的發現附法規依據，以 path dependency 引用 |
