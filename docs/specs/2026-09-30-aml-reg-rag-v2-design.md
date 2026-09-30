# AML 法規問答 v2 設計文件：檢索管線

- 日期：2026-09-30
- 狀態：已實作（2026-09-30）
- 前一版：`2026-09-30-aml-reg-rag-design.md`（v1）

## 1. 目標

以 v1 baseline（README「Baseline」一節）為對照，提升檢索品質：

| baseline 發現 | 證據 | v2 對策 |
| ------------- | ---- | ------- |
| 情境題被金融機構版干擾 | c-007、c-010 的 Hit@1 錯誤，c-010 三次引用禁止條文 | LLM rerank（評分考慮適用對象）、hybrid |
| 跨法規題漏條文 | c-013 的預期條文不在 top-6，導致拒答 | query rewriting（拆解子查詢） |
| 自動生成題少數漏找 | 175 題中 6 題不在 top-6 | hybrid（BM25 對專有名詞敏感） |

三項技術各自可開關，以 eval 逐項量測貢獻（ablation），依結果決定預設值。

### 範圍

- 包含：hybrid 檢索（向量 + BM25，RRF 合併）、query rewriting（拆解子查詢）、LLM rerank、檢索管線開關、`reg eval compare`、檢索細節寫入 trace 與 eval 結果
- 不包含：英文版、自律規範 PDF、法規簡稱與舊名（會改變語料，無法與 baseline 對照）；`gemini-embedding-2` 與 `thinking_level` 比較；agentic 重新檢索、多輪對話、前端（v3）

### 完成標準

1. 以 `--retrieval-only` 依序執行 4 組設定：baseline（全關）、+hybrid、+hybrid+rerank、全開
2. 以 v1 設定與最佳設定各執行一次完整 eval（`--repeat 3`）
3. 結果以 `reg eval compare` 呈現並寫入 README「Ablation 結果」，依結果調整三個開關的預設值

## 2. 檢索管線

```
問題
 └─ rewrite（開關）→ 查詢清單 [原問題, 子查詢 ≤ 2]；關閉時為 [原問題]
     └─ 每個查詢：向量 top C ＋ BM25 top C（hybrid 開關）→ RRF
         └─ 各查詢結果 → RRF → 前 C 段候選
             └─ rerank（開關）→ 輕量模型逐段評分 → 前 k 段；關閉時依 RRF 取前 k 段
```

C 為 `RETRIEVAL_CANDIDATES`（預設 20），k 為 `TOP_K`。原問題永遠保留在查詢清單中，子查詢只增加候選、不取代原查詢，召回不會低於 v1。

**RRF**：每段分數為各排名中 `1 / (60 + 名次)` 的總和，以 chunk_id 去重。

### 2.1 模組（`retrieval/`）

| 模組 | 內容 |
| ---- | ---- |
| `tokenize.py` | NFKC 後，連續中文字切為重疊 bigram（單一中文字保留為一個 token），連續英數字（含 `.`、`-`）整段轉小寫保留 |
| `bm25.py` | `Bm25Index`：以 `rank-bm25`（`BM25Okapi`）在記憶體建立索引，文件為 `page_content`（含標頭）；`search(query, n)` 回傳段落與 BM25 分數 |
| `fusion.py` | `reciprocal_rank_fusion(rankings)` |
| `rewrite.py` | `QueryRewriter`：`LLM_LIGHT_MODEL` 以 structured output 產生 0～2 個子查詢；程式把原問題放第一、去重、上限 3 個 |
| `rerank.py` | `LlmReranker`：候選標為 `C1`…`Cn`，輕量模型逐段給 0～3 分；依分數排序，同分依 RRF 名次 |
| `pipeline.py` | `Retriever` Protocol 與 `RetrievalPipeline` |
| `index.py` | 新增 `load_chunks()` 讀出索引表全部段落，供建立 BM25 |

rewrite 與 rerank 的 prompt 屬於檢索邏輯，放在各自模組內。

**rerank 評分**

| 分數 | 意義 |
| ---- | ---- |
| 3 | 直接回答問題，且適用對象正確 |
| 2 | 相關，但非完整答案 |
| 1 | 部分相關 |
| 0 | 無關，或適用對象不同（例如問虛擬資產服務商卻是金融機構的規定） |

### 2.2 介面

- `Retriever` Protocol：`retrieve(question, k) -> Retrieval`
- `Retrieval`：`chunks`（最終段落）與檢索細節：
  - `queries`：實際使用的查詢
  - `traces`：每個候選段落的 `ChunkTrace`（`chunk_id`、向量名次與相似度、BM25 名次與分數、RRF 分數、rerank 分數、最終名次）
  - `rewrite_fallback`、`rerank_fallback`：是否退回，與錯誤描述（例外類型與狀態碼）
  - `input_tokens`、`output_tokens`：輕量模型用量
- `RetrievedChunk.score` 改為最終排序依據：rerank 開啟時為 rerank 分數，否則為 RRF 分數；向量相似度保留在 `ChunkTrace`
- `PgChunkIndex.search()` 保留為純向量檢索，由管線呼叫
- graph 的 retrieve 節點、eval runner、`reg search` 都改用 `Retriever`；`AskState` 新增 `retrieval`

### 2.3 失敗處理

| 情況 | 處理 |
| ---- | ---- |
| rewrite 呼叫或解析失敗 | 查詢清單只用原問題，標記 `rewrite_fallback` |
| rerank 呼叫或解析失敗 | 依 RRF 名次，標記 `rerank_fallback` |
| rerank 輸出缺部分標籤 | 缺的段落視為 0 分；不存在的標籤忽略 |
| BM25 建立失敗 | 視為錯誤（與向量索引同一張表，代表資料庫有問題） |

退回不中斷問答，記錄於 trace 與 eval 結果。

## 3. 設定

| 設定 | 預設值 | 說明 |
| ---- | ------ | ---- |
| `RETRIEVAL_HYBRID` | `true` | 加入 BM25 |
| `RETRIEVAL_REWRITE` | `false` | 拆解子查詢（依 ablation 結果改為預設關閉，見 README） |
| `RETRIEVAL_RERANK` | `true` | LLM rerank |
| `RETRIEVAL_CANDIDATES` | `20` | 各檢索器取回段數與 rerank 候選數，須 ≥ `TOP_K`，否則啟動時報設定錯誤 |

rewrite 與 rerank 使用 `LLM_LIGHT_MODEL`，temperature 維持預設。ablation 後若某項無助益，改其預設為 `false` 並於 README 說明。

## 4. CLI

`search`、`ask`、`eval run` 新增 `--hybrid/--no-hybrid`、`--rewrite/--no-rewrite`、`--rerank/--no-rerank`，覆寫設定。

| 指令 | 變更 |
| ---- | ---- |
| `reg search` | 先列出查詢清單；表格增加向量名次、BM25 名次、rerank 分數；退回時顯示警告 |
| `reg ask --show-context` | 同上 |
| `reg eval compare <結果檔>...` | 新增：各結果一欄，列出各題庫與類別的 Hit@1、Hit@3、Recall@k、MRR；有生成結果時加上驗證通過率、拒答正確率、要點涵蓋率、成本；欄名為檢索設定（如 `hybrid+rerank`） |

## 5. Eval 與 trace

- eval 結果的 `settings` 加入三個開關與 `RETRIEVAL_CANDIDATES`
- 每筆檢索結果加入查詢數、是否退回、輕量模型 token
- 檢索摘要加入退回次數、平均耗時、成本（以 `LLM_LIGHT_MODEL` 計價）
- 生成成本加入檢索階段的輕量模型 token
- trace 加入 `retrieval`（查詢清單、各段 `ChunkTrace`、退回、token）

ablation 流程（寫入 README）：

```bash
uv run reg eval run --retrieval-only --no-hybrid --no-rewrite --no-rerank
uv run reg eval run --retrieval-only --no-rewrite --no-rerank
uv run reg eval run --retrieval-only --no-rewrite
uv run reg eval run --retrieval-only
uv run reg eval compare var/eval/<四個結果檔>
```

檢索 eval 各執行一次；rewrite 與 rerank 有隨機性，檢索的 `--repeat` 留待後續。

**成本估計**：全開時每次問答多 2 次輕量模型呼叫，約 0.3 美分、2～3 秒；一次檢索 eval（192 題）約 0.6 美元。

## 6. 測試

- 單元測試（無資料庫、無 API）：
  - tokenize：中文 bigram、英數保留、全半形、單字
  - fusion：名次換分、去重、同分順序
  - bm25：以 fixture 條文建立索引，專有名詞查詢（如「臨時性交易」）排第一
  - rewrite：原問題在首、去重、上限、失敗退回
  - rerank：排序、同分依 RRF、缺標籤為 0、多餘標籤忽略、失敗退回
  - pipeline：fake 向量索引、BM25 與 LLM，涵蓋 8 種開關組合與檢索細節
  - graph、runner、trace、cli：改用 `Retriever` 後行為不變，新開關與 `eval compare` 輸出
- `@pytest.mark.integration`：真實 Postgres 建立 BM25；真實輕量模型執行 rewrite 與 rerank 各一次

## 7. 文件

- README 新增「v2 檢索管線」與「Ablation 結果」，保留「Baseline」
- AGENTS.md 註明 rewrite 與 rerank 的 prompt 位置
- 說明 v1 trace 的 `top_score` 為向量相似度，v2 起 `score` 為最終排序分數
- 根目錄專案索引改為「v2 完成」
- v1 與 v2 的學習文件（根目錄 `docs/aml_reg_rag/`）於 v2 完成後一併撰寫，不在本 spec 範圍
