# AML 法規問答：虛擬資產防制洗錢法規 RAG

輸入自然語言問題，從台灣防制洗錢相關法規中檢索條文，產出每個論點都附條號與原文引述的回答。引用是否屬實由程式驗證，條文不足以回答時拒答。

> 輸出僅供查詢法規依據，不構成法律意見。問題會送到 LLM 與 embedding API，請勿輸入客戶個資或案件機敏資訊。

## 範圍

| 項目 | 說明 |
| ---- | ---- |
| 語料 | 6 部現行法規中文版（全國法規資料庫 Open API），清單見 `data/corpus.toml` |
| 檢索 | 向量檢索（Gemini embedding + pgvector cosine）、BM25、query rewriting、LLM rerank，各自可開關 |
| 生成 | 結構化輸出，每個論點至少一個引用與逐字引述，程式驗證 |
| 不支援 | 法規簡稱與舊名、英文版、自律規範、歷史版本、多輪對話 |

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
| retrieve | 程式與輕量模型 | 經檢索管線（見下節）取 `TOP_K` 段，依序標為 `S1`…`Sk` |
| generate | LLM | 只依段落作答，輸出論點、引用標籤與逐字引述；不足以回答時拒答並說明缺少什麼 |
| verify | 程式 | 引用來源須在本次檢索結果中，引述經 NFKC 正規化、移除空白後須為段落內文的子字串且至少 8 字 |

## 檢索管線

```
問題
 └─ rewrite → [原問題, 子查詢 ≤ 2]
     └─ 每個查詢：向量 top C ＋ BM25 top C → RRF
         └─ 各查詢結果 → RRF → 前 C 段候選
             └─ rerank → 輕量模型逐段評分 0～3 → 前 k 段
```

| 步驟 | 設定 | 內容 |
| ---- | ---- | ---- |
| hybrid | `RETRIEVAL_HYBRID` | BM25 以中文字元 bigram 斷詞，與向量檢索的名次以 RRF（`1 / (60 + 名次)`）合併 |
| rewrite | `RETRIEVAL_REWRITE` | 輕量模型把涉及多個議題的問題拆成子查詢；原問題永遠保留，召回不低於只用原問題 |
| rerank | `RETRIEVAL_RERANK` | 輕量模型依相關程度與適用對象評分，同分時依 RRF 名次 |

- C 為 `RETRIEVAL_CANDIDATES`（預設 20），k 為 `TOP_K`；rewrite 與 rerank 使用 `LLM_LIGHT_MODEL`
- rewrite 或 rerank 失敗時退回（只用原問題、依 RRF 名次），不中斷問答，並記錄於 trace 與 eval 結果
- `RetrievedChunk.score` 為最終排序依據：開啟 rerank 時為 rerank 分數，否則為 RRF 分數；向量相似度記錄在 trace 的 `retrieval.traces`。v1 的 trace 與 eval 結果中 `top_score` 為向量相似度，與 v2 不可直接比較

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
uv run reg search "臨時性交易多少金額要確認客戶身分"     # 只檢索；開啟 rewrite 或 rerank 時會呼叫輕量模型
uv run reg search "..." --no-rerank --no-rewrite        # 關閉部分檢索步驟（search、ask、eval run 皆可用）
uv run reg ask "虛擬資產服務商的交易紀錄要保存多久？"     # 問答
uv run reg ask "..." --show-context                    # 一併顯示檢索到的段落
uv run reg eval generate                               # 產生自動生成題（已存在時需 --overwrite）
uv run reg eval run --retrieval-only                   # 只評估檢索
uv run reg eval run                                    # 檢索 + 生成
uv run reg eval run --model google_genai:gemini-3.5-flash-lite --repeat 3
uv run reg eval compare var/eval/<結果檔>...            # 並排比較多次 eval
```

`reg ask` 在引用驗證失敗時以 exit code 1 結束；條文不足而拒答屬正常結果，exit code 為 0。

產出：

| 路徑 | 內容 |
| ---- | ---- |
| `var/raw/` | 全國法規資料庫下載快取 |
| `var/traces/<日期>.jsonl` | 每次 `ask` 的問題、檢索結果與分數、檢索細節（查詢、各階段名次、退回）、回答、驗證結果、token、耗時 |
| `var/eval/<時間>-<檢索設定>-<模型>.json` | eval 設定、索引資訊、各指標與每題明細；同一秒產生多份時加序號 |

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

v1 第一次 eval（2026-09-30，`reg eval run --repeat 3`）：`LLM_MODEL` 與 judge 皆為 `google_genai:gemini-3.8-flash`，`EMBEDDING_MODEL=google_genai:gemini-embedding-001`（1536 維），索引 `chunks_google_genai_gemini_embedding_001_d1536`（214 段），k=6。人工題 20 題皆為 `claude_draft`、尚未審核。

檢索（以條為單位）：

| 題庫 | 題數 | Hit@1 | Hit@3 | Hit@k | Recall@k | MRR |
| ---- | ---- | ----- | ----- | ----- | -------- | --- |
| curated | 17 | 76% | 94% | 94% | 91% | 0.853 |
| curated/lookup | 5 | 80% | 100% | 100% | 100% | 0.900 |
| curated/scenario | 5 | 60% | 100% | 100% | 90% | 0.800 |
| curated/cross_law | 3 | 67% | 67% | 67% | 67% | 0.667 |
| curated/distractor | 4 | 100% | 100% | 100% | 100% | 1.000 |
| synthetic | 175 | 74% | 93% | 97% | 97% | 0.834 |

生成（人工題 × 3 次）：

| 類別 | 執行 | 驗證通過 | 拒答正確 | 該答卻拒 | 該拒卻答 | 引用預期 | 引用禁止 | 要點涵蓋 | 平均秒數 | 成本 |
| ---- | ---- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | ---- |
| all | 60 | 100% | 95% | 3 | 0 | 96% | 3 | 88% | 4.8 | $0.3478 |
| lookup | 15 | 100% | 100% | 0 | 0 | 87% | 0 | 96% | 4.2 | $0.0775 |
| scenario | 15 | 100% | 100% | 0 | 0 | 100% | 3 | 80% | 5.8 | $0.1163 |
| cross_law | 9 | 100% | 67% | 3 | 0 | 100% | 0 | 88% | 5.8 | $0.0581 |
| distractor | 12 | 100% | 100% | 0 | 0 | 100% | 0 | 88% | 4.5 | $0.0664 |
| refusal | 9 | 100% | 100% | 0 | 0 | — | 0 | — | 3.8 | $0.0294 |

重複執行一致率 100%，無錯誤、無無法解析的輸出。

觀察（v2 改進方向）：

- c-013（未通報制裁資產的罰則）檢索未取得資恐防制法第 12 條與洗錢防制法第 5 條，3 次皆拒答；跨法規題需要多步檢索或 query rewriting
- 以情境描述的問題較容易被金融機構版辦法干擾：c-007、c-010 的第 1 名為金融機構防制洗錢辦法，c-010 三次都引用了禁止條文 `G0380252:15`；純條文查詢的干擾題則全部正確
- 自動生成題 175 題中有 6 題未在 top-6 找到出處條文

## Ablation 結果

2026-09-30 執行；`LLM_MODEL` 與 judge 為 `google_genai:gemini-3.8-flash`，`LLM_LIGHT_MODEL` 為 `google_genai:gemini-3.5-flash-lite`，索引沿用 v1 的 `chunks_google_genai_gemini_embedding_001_d1536`（214 段），k=6、`RETRIEVAL_CANDIDATES=20`。除依序疊加的 4 組外，另補跑只開 rerank 的一組，用來分離 hybrid 在搭配 rerank 時的影響。

```bash
uv run reg eval run --retrieval-only --no-hybrid --no-rewrite --no-rerank
uv run reg eval run --retrieval-only --no-rewrite --no-rerank
uv run reg eval run --retrieval-only --no-rewrite
uv run reg eval run --retrieval-only --hybrid --rewrite --rerank
uv run reg eval run --retrieval-only --no-hybrid --no-rewrite --rerank
uv run reg eval compare var/eval/<以上結果檔>
```

檢索（以條為單位；vector 一欄與「Baseline」一致）：

| 題庫 | 指標 | vector（v1） | hybrid | rerank | hybrid+rerank | hybrid+rewrite+rerank |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| curated | Hit@1 | 76% | 76% | 82% | 88% | 82% |
| curated | Hit@3 | 94% | 88% | 100% | 100% | 100% |
| curated | Recall@k | 91% | 85% | 94% | 94% | 94% |
| curated | MRR | 0.853 | 0.824 | 0.912 | 0.941 | 0.912 |
| curated/lookup | Hit@1 | 80% | 80% | 80% | 80% | 80% |
| curated/lookup | Hit@3 | 100% | 80% | 100% | 100% | 100% |
| curated/lookup | Recall@k | 100% | 80% | 100% | 100% | 100% |
| curated/lookup | MRR | 0.900 | 0.800 | 0.900 | 0.900 | 0.900 |
| curated/scenario | Hit@1 | 60% | 80% | 60% | 80% | 60% |
| curated/scenario | Hit@3 | 100% | 100% | 100% | 100% | 100% |
| curated/scenario | Recall@k | 90% | 90% | 90% | 90% | 90% |
| curated/scenario | MRR | 0.800 | 0.900 | 0.800 | 0.900 | 0.800 |
| curated/cross_law | Hit@1 | 67% | 33% | 100% | 100% | 100% |
| curated/cross_law | Hit@3 | 67% | 67% | 100% | 100% | 100% |
| curated/cross_law | Recall@k | 67% | 67% | 83% | 83% | 83% |
| curated/cross_law | MRR | 0.667 | 0.500 | 1.000 | 1.000 | 1.000 |
| curated/distractor | Hit@1 | 100% | 100% | 100% | 100% | 100% |
| curated/distractor | Hit@3 | 100% | 100% | 100% | 100% | 100% |
| curated/distractor | Recall@k | 100% | 100% | 100% | 100% | 100% |
| curated/distractor | MRR | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| synthetic | Hit@1 | 74% | 72% | 79% | 79% | 78% |
| synthetic | Hit@3 | 93% | 90% | 95% | 95% | 95% |
| synthetic | Recall@k | 97% | 96% | 97% | 96% | 95% |
| synthetic | MRR | 0.834 | 0.816 | 0.870 | 0.869 | 0.859 |

| 指標 | vector（v1） | hybrid | rerank | hybrid+rerank | hybrid+rewrite+rerank |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 每題平均秒數（synthetic） | 0.4 | 0.4 | 2.5 | 2.9 | 4.0 |
| 一次檢索 eval 成本（192 題） | $0.00 | $0.00 | $0.54 | $0.55 | $0.56 |
| 退回次數 | 0 | 0 | 0 | 0 | 0 |

完整 eval（人工題 × 3 次，`--repeat 3`）：v1 設定（`--no-hybrid --no-rewrite --no-rerank`）與最佳設定（hybrid+rerank）：

| 類別 | 設定 | 驗證通過 | 拒答正確 | 該答卻拒 | 該拒卻答 | 引用預期 | 引用禁止 | 要點涵蓋 | 平均秒數 | 成本 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| all | vector | 100% | 95% | 3 | 0 | 94% | 3 | 90% | 4.7 | $0.3544 |
| all | hybrid+rerank | 100% | 98% | 1 | 0 | 96% | 3 | 92% | 7.7 | $0.5954 |
| lookup | vector | 100% | 100% | 0 | 0 | 80% | 0 | 93% | 3.8 | $0.0747 |
| lookup | hybrid+rerank | 100% | 100% | 0 | 0 | 87% | 0 | 96% | 7.1 | $0.1356 |
| scenario | vector | 100% | 100% | 0 | 0 | 100% | 3 | 83% | 6.1 | $0.1217 |
| scenario | hybrid+rerank | 100% | 100% | 0 | 0 | 100% | 3 | 96% | 8.5 | $0.1828 |
| cross_law | vector | 100% | 67% | 3 | 0 | 100% | 0 | 88% | 5.7 | $0.0634 |
| cross_law | hybrid+rerank | 100% | 89% | 1 | 0 | 100% | 0 | 75% | 12.2 | $0.1306 |
| distractor | vector | 100% | 100% | 0 | 0 | 100% | 0 | 96% | 4.1 | $0.0652 |
| distractor | hybrid+rerank | 100% | 100% | 0 | 0 | 100% | 0 | 96% | 6.1 | $0.0955 |
| refusal | vector | 100% | 100% | 0 | 0 | — | 0 | — | 3.6 | $0.0294 |
| refusal | hybrid+rerank | 100% | 100% | 0 | 0 | — | 0 | — | 5.3 | $0.0509 |

一致率：vector 100%、hybrid+rerank 95%（不穩定：c-013）

### 結論與預設值

- **rerank 是主要改善來源**：curated Hit@1 由 76% 提升到 82%～88%，synthetic Hit@1 由 74% 提升到 79%、MRR 由 0.834 提升到 0.870；cross_law 的 Hit@1 由 67% 提升到 100%
- **hybrid 單獨使用略差**（curated Hit@3 94% → 88%），搭配 rerank 時在 synthetic 上與只開 rerank 持平，curated 多對 1 題情境題；BM25 在記憶體中執行，幾乎不增加耗時與成本，預設保留。差距只有 1 題，可能是 rerank 的隨機性
- **rewrite 沒有提升檢索**（curated Hit@1 88% → 82%、synthetic MRR 0.869 → 0.859），每題多約 1.1 秒；同一題（c-013）在兩次執行中拆出的子查詢不同，一次取得資恐防制法第 12 條、一次沒有，結果不穩定。**預設改為關閉**（`RETRIEVAL_REWRITE=false`）
- 預設設定為 hybrid+rerank；檢索每題多約 2.5 秒，檢索 eval（192 題）約 0.55 美元

### 觀察

- c-013（未通報制裁資產的罰則）：v1 三次都拒答；v2 檢索取得資恐防制法第 12 條，三次中兩次回答、一次拒答（一致率因此由 100% 降為 95%）。回答時仍缺「洗錢防制法第 5 條：虛擬資產服務商適用金融機構規定」這個要點，要點涵蓋率 50%，因此 cross_law 的要點涵蓋率反而由 88% 降為 75%（v1 拒答的執行不計入涵蓋率）
- c-010（未完成的疑似洗錢交易）：兩種設定都三次引用禁止條文金融機構防制洗錢辦法第 15 條。題目沒有提到虛擬資產服務商，rerank 將第 15 條評為相關是合理的；這題的禁止條文設定需要在審核題庫時重新檢視
- 完整 eval 每題平均耗時由 4.7 秒增為 7.7 秒，成本由 0.35 美元增為 0.60 美元（+68%），換得拒答正確率 95% → 98%、要點涵蓋率 90% → 92%、引用預期條文 94% → 96%
- 第一次執行全開設定時撞到 Gemini 專案的每月支出上限（後 98 題回傳 429），調高上限後重跑，上表為重跑結果
