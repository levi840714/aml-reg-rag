# AML 法規問答：虛擬資產防制洗錢法規 RAG

輸入自然語言問題，從台灣防制洗錢相關法規中檢索條文，產出每個論點都附條號與原文引述的回答。引用是否屬實由程式驗證，條文不足以回答時拒答。

> 輸出僅供查詢法規依據，不構成法律意見。問題會送到 LLM 與 embedding API，請勿輸入客戶個資或案件機敏資訊。

## 範圍

| 項目 | 說明 |
| ---- | ---- |
| 語料 | 6 部現行法規中文版（全國法規資料庫 Open API），清單見 `data/corpus.toml` |
| 檢索 | 向量檢索（Gemini embedding + pgvector cosine） |
| 生成 | 結構化輸出，每個論點至少一個引用與逐字引述，程式驗證 |
| 不支援 | BM25 / hybrid、reranker、法規簡稱與舊名、英文版、自律規範、歷史版本、多輪對話 |

語料：洗錢防制法、資恐防制法、提供虛擬資產服務之事業或人員防制洗錢及打擊資恐辦法、提供虛擬資產服務之事業或人員洗錢防制登記辦法、金融機構及提供虛擬資產服務之事業或人員防制詐欺犯罪危害應遵循事項辦法、金融機構防制洗錢辦法（與 VASP 辦法內容相近，作為檢索干擾項）。

## 流程

```
匯入：fetch ──→ parse ──→ chunk ──→ embed + store

問答：retrieve ──→ generate ──→ verify ──(通過)────────────→ 結束
                     ↑            │
                     └─(未通過，重試 1 次，附問題清單)
                                  └─(重試後仍未通過)──→ 結束（標記驗證失敗）
```

| 階段 | 負責 | 內容 |
| ---- | ---- | ---- |
| chunk | 程式 | 依項、款、目切段（上限 `CHUNK_MAX_CHARS`），切到款或目時帶上上層引言；每段標頭含法規名稱與條號，與內文一起 embedding |
| retrieve | 程式 | 取 `TOP_K` 段，依序標為 `S1`…`Sk` |
| generate | LLM | 只依段落作答，輸出論點、引用標籤與逐字引述；不足以回答時拒答並說明缺少什麼 |
| verify | 程式 | 引用來源須在本次檢索結果中，引述經 NFKC 正規化、移除空白後須為段落內文的子字串且至少 8 字 |

## 安裝

```bash
uv sync
cp .env.example .env        # 填入 GOOGLE_API_KEY
docker compose up -d        # 或將 DATABASE_URL 指向已安裝 pgvector 的本機 Postgres
uv run reg ingest
```

- `reg ingest` 會刪除並重建自己的索引表（`chunks_*` 與 `index_meta` 中對應的列），`DATABASE_URL` 不要指向共用、測試或正式環境的資料庫；指向非本機主機時需加 `--allow-remote` 才會執行
- `.env` 設定 `LANGSMITH_TRACING=true` 時，問題與回答會送到 LangSmith

## 使用

```bash
uv run reg search "臨時性交易多少金額要確認客戶身分"     # 只檢索，不呼叫 LLM
uv run reg ask "虛擬資產服務商的交易紀錄要保存多久？"     # 問答
uv run reg ask "..." --show-context                    # 一併顯示檢索到的段落
uv run reg eval generate                               # 產生自動生成題（已存在時需 --overwrite）
uv run reg eval run --retrieval-only                   # 只評估檢索
uv run reg eval run                                    # 檢索 + 生成
uv run reg eval run --model google_genai:gemini-3.5-flash-lite --repeat 3
```

`reg ask` 在引用驗證失敗時以 exit code 1 結束；條文不足而拒答屬正常結果，exit code 為 0。

產出：

| 路徑 | 內容 |
| ---- | ---- |
| `var/raw/` | 全國法規資料庫下載快取 |
| `var/traces/<日期>.jsonl` | 每次 `ask` 的問題、檢索結果與分數、回答、驗證結果、token、耗時 |
| `var/eval/<時間>-<模型>.json` | eval 設定、索引資訊、各指標與每題明細 |

## Eval

| 題庫 | 用途 |
| ---- | ---- |
| `eval/curated.jsonl` | 人工題：條文查詢、情境判斷、跨法規、干擾辨識、應拒答；評估檢索與生成 |
| `eval/synthetic.jsonl` | 每條條文由輕量模型產生一題；只評估檢索 |

- 檢索：Hit@1、Hit@3、Hit@k、Recall@k、MRR（以條為單位）
- 生成：驗證通過率、拒答判斷（該答卻拒、該拒卻答）、是否引用預期或禁止條文、要點涵蓋率（`EVAL_JUDGE_MODEL` 判斷）、成本與耗時
- `--repeat N` 每題執行 N 次，列出拒答判斷或驗證結果不一致的題目

比較 embedding 模型、維度或切段大小時，修改 `.env` 後重新 `reg ingest` 再執行 eval。

## Baseline

v1 第一次 eval 的結果（`reg eval run`，預設設定）：

尚未執行。建立索引並產生自動生成題後執行 `uv run reg eval run --repeat 3`，將檢索與生成摘要表格貼於此處，並註明執行日期、模型與索引表名稱。
