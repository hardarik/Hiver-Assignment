# Customer Support AI Agent & Evaluation Harness (@AmazonHelp)

A production-focused AI Customer Support Agent for **`@AmazonHelp`** built on real-world Twitter support conversations. Features multi-intent classification, historical RAG grounding, an operational triage/escalation decision engine, and a complete evaluation harness benchmarking headline results against two baselines.

---

## ⚡ Quickstart (Reproduce Headline Results in < 15 Seconds)

### 1. Installation
```bash
# Clone repo & install requirements
pip install -r requirements.txt
```

### 2. Run Benchmark Evaluation
```bash
python run_pipeline.py --eval
```
*Outputs accuracy, Macro/Weighted F1, Escalation F1, reply quality rubric scores, and Cohen's Kappa judge-human agreement metrics on 200 hand-labeled golden examples.*

### 3. Run Interactive Demo
```bash
python run_pipeline.py --demo "My Fire TV stick is buffering continuously!"
```

### 4. Run Test Suite
```bash
pytest tests/
```

---

## 📊 Headline Benchmark Results (200 Golden Examples)

| System Tier | Intent Accuracy | Intent F1 (Macro) | Escalation F1 | Reply Quality Score (1-5) |
|---|---|---|---|---|
| **Trivial Baseline** | 20.0% | 0.0476 | 0.0000 | 4.01 / 5.0 |
| **Simple Baseline** | 64.0% | 0.5974 | 0.4800 | 4.14 / 5.0 |
| **Proposed System** | **96.5%** | **0.9683** | **0.9867** | **4.77 / 5.0** |

* **LLM Judge vs Human Alignment Evidence**: **96.0% Agreement**, **Cohen's Kappa = 0.884** (*Substantial Agreement* over 50 hand-annotated validation items).

---

## 📁 Repository Folder & Component Design

```
.
├── data/
│   ├── twcs_sample.csv               # Raw TWCS Twitter dataset sample
│   ├── amazon_help_pairs.parquet     # Processed @AmazonHelp pairs (20,644 rows)
│   ├── golden_eval_set.json          # 200 Hand-Labeled Golden Examples
│   ├── golden_eval_set_note.md       # Sampling & Labeling methodology notes
│   └── results_benchmark.json        # Output benchmark metrics JSON
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Pair extraction & ASCII text cleaning
│   ├── intent_classifier.py          # 7-Class hybrid ML + Rule Intent Classifier
│   ├── rag_retriever.py              # Grounding index over @AmazonHelp resolutions
│   ├── escalation_engine.py          # Risk, actionability & anger triage matrix
│   ├── agent.py                      # Master AmazonSupportAgent pipeline
│   ├── baselines.py                  # Trivial and Simple baseline implementations
│   ├── llm_judge.py                  # 5-Dimension rubric evaluator & Kappa validation
│   └── eval_harness.py               # Evaluation benchmark execution harness
├── tests/
│   ├── __init__.py
│   └── test_agent.py                 # Pytest suite
├── scripts/
│   └── build_golden_dataset.py       # Golden Dataset generation script
├── run_pipeline.py                   # Master CLI entry point
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation & Quickstart
├── REPORT.md                         # 5-Section Technical Assignment Report
└── DECISION_LOG.md                   # 12 Non-Obvious Engineering Decisions
```

---

## 🛠️ Human (Engineer) vs. AI Assistant Contribution Breakdown

Per assignment rules (*"You may use AI coding assistants freely... Cite anything you borrowed"*), here is the transparent breakdown of human engineering decisions vs AI-assisted code generation:

### 🧠 Human (Candidate / Engineer) Driven Work:
1. **Problem Framing & Scope Boundaries**: Defined operational boundaries for `@AmazonHelp` (what to auto-handle vs escalate; explicit refusal to execute financial transactions or DM auth via Twitter public tweets).
2. **Intent Taxonomy Architecture**: Designed the 7 domain-specific intents tailored to e-commerce customer support instead of over-segmented 77-class generic taxonomies.
3. **Golden Evaluation Set Curation**: Formulated the sampling strategy, hand-labeled all 200 golden items, defined escalation criteria, and drafted ground-truth policy rationales.
4. **12 Non-Obvious Engineering Decisions**: Authored the rationale behind high-confidence rule overrides, stratified sampling, ASCII filtering, and local evaluation harnesses in `DECISION_LOG.md`.
5. **Failure Analysis & Headline Limitations**: Analyzed top 5 real failure modes (sarcasm, compound intents, expired short URLs) and authored the mandatory critique on why headline metrics can be misleading.

### 🤖 AI Coding Assistant Accelerated Work:
1. **Boilerplate & Helper Script Automation**: Generated pandas data processing logic in `data_loader.py` and dataset conversion scripts in `scripts/build_golden_dataset.py`.
2. **Scikit-Learn Pipeline Wiring**: Assisted with standard TF-IDF vectorizer and Naive Bayes pipeline configuration in `intent_classifier.py`.
3. **Evaluation Harness Metrics Wiring**: Generated `scikit-learn` precision/recall/F1 calculation and `rich` CLI table formatting in `eval_harness.py` and `run_pipeline.py`.
4. **Pytest Boilerplate**: Generated standard test fixture stubs in `tests/test_agent.py`.

---

## 🔗 Assignment Deliverables Quick Links
- 📜 [REPORT.md](file:///c:/Users/ariks/New%20folder/REPORT.md) — Technical report covering Problem Framing, Baselines, Failure Modes, Headline Critique, and 1-Week Roadmap.
- 📋 [DECISION_LOG.md](file:///c:/Users/ariks/New%20folder/DECISION_LOG.md) — 12 non-obvious engineering decisions.
- 📝 [data/golden_eval_set_note.md](file:///c:/Users/ariks/New%20folder/data/golden_eval_set_note.md) — Sampling and hand-labeling documentation.
