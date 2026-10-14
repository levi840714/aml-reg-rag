# aml-reg-rag：台灣防制洗錢法規問答

針對虛擬資產服務商（VASP）的防制洗錢法規問答 CLI。輸入自然語言問題，從台灣 6 部防制洗錢相關法規中檢索條文，回答的每個論點都附上條號、原文引述與全國法規資料庫連結。

- **引用由程式驗證**：引述必須逐字出自檢索到的段落，不符合時要求模型重寫一次，仍不符合就標記驗證失敗
- **條文不足就拒答**：說明缺少哪類規範，不以段落以外的知識補足
- **依法規結構切段**：依項、款、目切分，並帶上上層引言，避免條文被切斷
- **可開關的檢索管線**：向量檢索、BM25、query rewriting、LLM rerank，各步驟的效果以 eval 量測
- **沿條文引用補段落**：解析「準用」「前項」「第 X 條」等引用，檢索後補上被準用、被引用的條文與對應罰則
- **兩套題庫的 eval**：人工題評估檢索與回答，自動生成題評估檢索，可並排比較不同設定

> 輸出僅供查詢法規依據，不構成法律意見。問題會送到 LLM 與 embedding API，請勿輸入客戶個資或案件機敏資訊。

## 輸出範例

```console
$ uv run reg ask "虛擬資產服務商的交易紀錄要保存多久？"
1. 虛擬資產服務商對國內外交易之所有必要紀錄，應至少保存五年；但法律另有較長保存期間規定者，從其規定。 [1]
   「對國內外交易之所有必要紀錄，應至少保存五年。但法律另有較長保存期間規定者，從其規定。」
2. 虛擬資產交易平台之相關交易紀錄，自交易完成時起應保存五年；若遇有爭議，則應保存至爭議消除為止。 [2]
   「前項交易紀錄之保存，自交易完成時起，應保存五年。但遇有爭議者，應保存至爭議消除為止。」

來源：
  [1] 提供虛擬資產服務之事業或人員防制洗錢及打擊資恐辦法 第 10 條 https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode=G0380268&flno=10
  [2] 提供虛擬資產服務之事業或人員洗錢防制登記辦法 第 20 條 https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode=G0400162&flno=20
```

條文不足以回答時（例如問虛擬資產的課稅方式），會顯示「條文不足以回答」、缺少的規範，以及檢索到的條文清單。

## 快速開始

需求：Python 3.12 以上、[uv](https://docs.astral.sh/uv/)、Docker（或已安裝 pgvector 的 Postgres）、Gemini API key。建立索引與執行 eval 的呼叫量超過 Gemini 免費額度，需使用付費方案。

```bash
uv sync
cp .env.example .env        # 填入 GOOGLE_API_KEY
docker compose up -d        # pgvector，綁定 127.0.0.1:55432
uv run reg ingest           # 下載法規、切段、建立索引
uv run reg ask "虛擬資產服務商辦理臨時性交易，多少金額要確認客戶身分？"
```

- `reg ingest` 會刪除並重建自己的索引表，`DATABASE_URL` 不要指向共用、測試或正式環境的資料庫；指向非本機主機時需加 `--allow-remote` 才會執行
- `.env` 設定 `LANGSMITH_TRACING=true` 時，問題與回答會送到 LangSmith

## 使用

```bash
uv run reg search "臨時性交易多少金額要確認客戶身分"       # 只看檢索結果與各階段名次
uv run reg ask "…"                                       # 問答
uv run reg ask "…" --show-context                        # 一併顯示檢索到的段落
uv run reg ask "…" --rewrite --no-rerank                 # 臨時切換檢索步驟（search、ask、eval run 皆可用）
uv run reg eval generate                                 # 產生自動生成題（已存在時需 --overwrite）
uv run reg eval run --retrieval-only                     # 只評估檢索
uv run reg eval run --repeat 3                           # 檢索加回答，每題執行 3 次
uv run reg eval run --model google_genai:gemini-3.5-flash-lite
uv run reg eval compare var/eval/<結果檔>...              # 並排比較多次 eval
```

`reg ask` 在引用驗證失敗時以 exit code 1 結束；條文不足而拒答是正常結果，exit code 為 0。

| 產出 | 內容 |
| ---- | ---- |
| `var/raw/` | 全國法規資料庫下載快取 |
| `var/traces/<日期>.jsonl` | 每次 `ask` 的問題、檢索細節（查詢、各階段名次與分數）、回答、驗證結果、token、耗時 |
| `var/eval/<時間>-<檢索設定>-<模型>.json` | eval 設定、索引資訊、各指標與每題明細 |

主要設定（完整清單見 `.env.example`）：

| 設定 | 預設值 | 說明 |
| ---- | ------ | ---- |
| `LLM_MODEL` | `google_genai:gemini-3.8-flash` | 回答生成 |
| `LLM_LIGHT_MODEL` | `google_genai:gemini-3.5-flash-lite` | rerank、query rewriting、自動生成題 |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `google_genai:gemini-embedding-001` / 1536 | 變更後需重新 `reg ingest` |
| `TOP_K` | 6 | 交給回答模型的段數 |
| `RETRIEVAL_HYBRID` / `RETRIEVAL_RERANK` / `RETRIEVAL_REWRITE` | `true` / `true` / `false` | 檢索步驟開關，見「檢索管線」 |
| `RETRIEVAL_EXPAND_REFS` / `RETRIEVAL_EXPAND_LIMIT` | `false` / 5 | 引用展開開關與最多補入段數，見「引用展開」 |

模型以 `<provider>:<model>` 指定，可換成 OpenAI 或 Anthropic 的模型比較。

## 架構

### 匯入

```
全國法規資料庫 Open API ──→ 解析 6 部法規 ──→ 依項、款、目切段 ──→ embedding ──→ pgvector
                                                    └──→ 解析條文引用 ──→ 引用關係表
```

- 法規清單在 `data/corpus.toml`：洗錢防制法、資恐防制法、提供虛擬資產服務之事業或人員防制洗錢及打擊資恐辦法、提供虛擬資產服務之事業或人員洗錢防制登記辦法、金融機構及提供虛擬資產服務之事業或人員防制詐欺犯罪危害應遵循事項辦法，以及內容相近、作為干擾項的金融機構防制洗錢辦法
- 整條不超過 `CHUNK_MAX_CHARS`（600 字）時為一段；超過時依項合併，單一項過長再依款、依目切分，並在每段前面帶上上層引言
- 每段標頭含法規名稱與條號，與內文一起 embedding，用來區分內容相近的金融機構版與 VASP 版辦法

### 檢索管線

```
問題
 └─ [rewrite] 拆成子查詢，原問題永遠保留
     └─ 每個查詢：向量檢索 top 20 ＋ [hybrid] BM25 top 20 → RRF 合併
         └─ 各查詢結果 → RRF → 前 20 段候選
             └─ [rerank] 輕量模型逐段評分 0～3 → 前 k 段
                 └─ [expand_refs] 沿引用關係補上最多 5 段
```

| 步驟 | 設定 | 預設 | 內容 |
| ---- | ---- | ---- | ---- |
| hybrid | `RETRIEVAL_HYBRID` | 開 | BM25 以中文字元 bigram 斷詞，與向量檢索的名次以 RRF（`1 / (60 + 名次)`）合併 |
| rerank | `RETRIEVAL_RERANK` | 開 | 輕量模型依相關程度與適用對象評分（例如問虛擬資產服務商時，金融機構的規定給 0 分），同分時依 RRF 名次 |
| rewrite | `RETRIEVAL_REWRITE` | 關 | 輕量模型把涉及多個議題的問題拆成子查詢；eval 中沒有提升檢索、每題多約 1 秒，因此預設關閉 |
| expand_refs | `RETRIEVAL_EXPAND_REFS` | 關 | 見「引用展開」；完整生成 eval 尚未跑完，暫不預設開啟 |

rewrite 或 rerank 失敗時退回（只用原問題、依 RRF 名次），不中斷問答，並記錄在 trace 與 eval 結果中。

### 引用展開

條文常以「準用第十六條」「前項第一款所稱」「違反第七條規定者，處…罰鍰」指向其他條文，只檢索到其中一條時，模型看不到完整規範。

- 匯入時以規則解析引用：中文數字條號、前項／前條／前 N 款、範圍（至）、`、及` 連接的同一法規沿用、「同法」「同條例」，以及辦法中「本法」依「以下簡稱本法」指向母法；語料外的法規（刑法、詐欺犯罪危害防制條例等）略過
- 每個引用依項、款、目對應到被引用的段落，並分類為準用、定義、罰則、授權、一般，寫入 `<索引表>_refs`
- 檢索後依名次逐段展開：準用與定義補上被引用的段落；罰則反向補上處罰該義務的條文；一般引用只補同一條內的項次（如「前項」）；授權依據不展開
- 補入的段落排在檢索結果之後，prompt 中標示為「補充：S1 準用的條文」等，引用驗證規則不變
- 兩端都已在結果中的引用也會標註（如第 16 條標示「S1 準用的條文」），並要求模型把被準用的規定套用到引用它的條文所規範的對象

已建立的索引需重新 `reg ingest` 才會有引用關係表。

### 回答與驗證

以 LangGraph 實作：

```
retrieve ──→ generate ──→ verify ──(通過)──────────────→ 結束
                ↑            │
                └─(未通過，重試 1 次，附上問題清單)
                             └─(重試後仍未通過)──→ 結束（標記驗證失敗）
```

- 段落依序標為 `S1`…`Sk`，模型以 structured output 回傳論點、引用標籤與逐字引述
- verify 由程式執行：引用的標籤須在本次檢索結果中，引述經 NFKC 正規化、移除空白後須為該段內文的子字串，且至少 8 個字
- 不另外產生總結：每句話都要能追溯到條文，總結無法由程式驗證

## 評估

| 題庫 | 題數 | 用途 |
| ---- | ---- | ---- |
| `eval/curated.jsonl` | 25 | 人工題：條文查詢、情境判斷、跨法規、需沿引用回答、干擾辨識、應拒答；評估檢索與回答 |
| `eval/synthetic.jsonl` | 175 | 每條條文由輕量模型產生一題；評估檢索 |

檢索以 Hit@1、Hit@3、Recall@k、MRR 衡量（以條為單位）；回答以驗證通過率、拒答判斷、是否引用正確或禁止的條文、要點涵蓋率（judge 模型判斷）衡量，並記錄耗時與成本。

目前結果（2026-09-30，`gemini-3.8-flash` 回答、`gemini-3.5-flash-lite` 輕量模型）：

| 指標 | 只有向量檢索 | 預設（hybrid + rerank） |
| ---- | ---- | ---- |
| 人工題檢索 Hit@1 | 76% | 88% |
| 人工題檢索 MRR | 0.853 | 0.941 |
| 自動生成題檢索 Hit@1 | 74% | 79% |
| 拒答判斷正確 | 95% | 98% |
| 該拒答卻回答 | 0 | 0 |
| 要點涵蓋 | 90% | 92% |
| 每題耗時（含檢索與回答） | 4.7 秒 | 7.7 秒 |

上表為 20 題人工題時的結果。2026-10-14 加入 5 題需沿引用回答的題目後，引用展開（上限 5 段）的檢索結果：人工題 Recall@k 92% → 98%、需沿引用回答的題目 85% → 100%，平均每題只多送 0.5 條；同樣多送條文的對照組（k=9）只到 93%。完整生成 eval（另加已檢索段落間的關係標註）：要點涵蓋 84% → 89%、需沿引用回答的題目 59% → 79%、該答卻拒 1 → 0，回答的 input token 多 24%；c-001 因展開帶入低相關條文而退步，細節見 eval 結果文件。

各步驟的 ablation、分類別數字、指標定義與個別題目的分析見 [docs/eval-results.md](docs/eval-results.md)。

比較不同設定時，以 CLI 參數或 `.env` 調整後重新執行 eval；embedding 模型、維度或切段大小變更後需先重新 `reg ingest`。

## 開發

```bash
uv run pytest                                   # 單元測試（fake model，不連資料庫與外部 API）
uv run pytest -m integration                    # 需要本機 pgvector；LLM 測試另需 API key 與已建立的索引
uv run ruff format src tests && uv run ruff check src tests && uv run mypy src
```

```
src/aml_reg_rag/
├── cli.py / render.py     # Typer 指令與 rich 顯示
├── ingest/                # 下載、解析、切段、引用解析、匯入
├── retrieval/             # embedding、pgvector 索引、BM25、RRF、rewrite、rerank、引用展開、檢索管線
├── graph/                 # LangGraph：retrieve → generate → verify
├── citations.py           # 引用驗證
└── evaluation/            # 題庫、自動生成題、judge、指標、比較
```

設計文件在 `docs/specs/`，實作計畫在 `docs/plans/`，開發慣例見 `AGENTS.md`。

## 限制

- 語料為 2026-09-18 版的 6 部法規中文版，不含法規簡稱與舊名對應、英文版、公會自律規範與歷史版本
- 單次問答，不支援多輪對話
- 人工題目前由 Claude 對照條文起草，尚未經法遵人員審核
- 引用驗證只確認引述出自條文，不保證答案的法律解釋正確

## 授權

[MIT](LICENSE)
