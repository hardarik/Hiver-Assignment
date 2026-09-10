# Technical Assignment Report: AI Customer Support Agent for @AmazonHelp

---

## 1. Problem Framing & Scope Definition

### What "Good" Means for `@AmazonHelp`
On Twitter, `@AmazonHelp` acts as a frontline triage and resolution handle. A high-trust AI agent for `@AmazonHelp` must excel at three operational goals:
1. **Precise Intent Categorization**: Instantly route incoming tweets into one of 7 clear intents (Shipping, Refunds, Damaged Goods, Account Security, Prime Video, Escalation Complaints, General Feedback).
2. **Grounded Brand Tone & Policy Resolution**: Draft concise, polite replies that align with `@AmazonHelp`'s historic resolution patterns, providing actual working links (`https://amzn.to/orders`, `https://amzn.to/help`).
3. **Strict Security & Risk Triage**: Know when **not** to auto-reply with standard advice. Transactions requiring account-level modifications (refunds, return authorizations, account cancellations, credit card updates) must be escalated to human agents via secure authenticated DM links.

### Out of Scope (What We Chose NOT to Build)
- **Autonomous Direct Refund Execution**: AI agents should not autonomously issue refunds or modify billing via public Twitter posts without human authentication.
- **Private DM Authentication Handshakes**: Real DM authentication requires OAuth integration with Amazon internal user databases, which is out of scope for a public dataset.
- **Multilingual Support Optimization**: While TWCS contains Japanese and German tweets, we focused our agent and evaluation benchmark on English customer support queries to maximize classification precision and evaluation reproducibility.

---

## 2. Results vs. Baselines

We evaluated the **Proposed System** against two baseline architectures on our **200 Hand-Labeled Golden Evaluation Set**:

- **Trivial Baseline**: Majority class predictor (`SHIPPING_DELIVERY_INQUIRY`), default auto-handle decision, and single static canned template reply.
- **Simple Baseline**: Keyword lookup intent classifier + heuristic triage matcher without RAG historical context.
- **Proposed System**: Hybrid Intent Classifier (TF-IDF/NB + High-Confidence Rule Overrides), Historical `@AmazonHelp` RAG Retriever (5,000 indexed resolution pairs), and Multi-Factor Triage Escalation Engine.

### Headline Benchmark Performance

| Performance Metric | Trivial Baseline | Simple Baseline | Proposed System | Delta (vs Simple) |
|---|---|---|---|---|
| **Intent Classification Accuracy** | 20.0% | 64.0% | **96.5%** | **+32.5%** |
| **Intent F1 Score (Macro)** | 0.0476 | 0.5974 | **0.9683** | **+0.3709** |
| **Intent F1 Score (Weighted)** | 0.0667 | 0.6346 | **0.9652** | **+0.3306** |
| **Escalation Precision** | 0.0000 | 1.0000 | **1.0000** | 0.0000 |
| **Escalation Recall** | 0.0000 | 0.3158 | **0.9737** | **+0.6579** |
| **Escalation F1 Score** | 0.0000 | 0.4800 | **0.9867** | **+0.5067** |
| **LLM Judge Reply Quality (1–5)** | 4.01 / 5.0 | 4.14 / 5.0 | **4.77 / 5.0** | **+0.63** |

### Judge-Human Agreement Validation
To validate our automated evaluation harness, we measured agreement between LLM Judge predictions and hand-annotated human labels on a 50-example validation subset:
- **Percent Agreement**: **96.0%**
- **Cohen's Kappa Score**: **0.884** (*Substantial Agreement*, Kappa $\ge 0.70$).

---

## 3. Failure Analysis: Top 5 Failure Modes

Through error auditing on test predictions, we identified five critical failure modes:

### Failure Mode 1: Compound Intent Queries (Damaged Item + Refund Request)
- **Real Example**: *"My Fire TV stick arrived today but the box was crushed and now I want a refund!"*
- **Ground Truth**: `DAMAGED_MISSING_ITEM` (`ESCALATE`).
- **Agent Prediction**: `REFUND_RETURN_REQUEST`.
- **Root Cause & Hypothesis**: Overlapping feature signals (`refund` vs `crushed/damaged`). The classifier triggered on the financial keyword rather than the physical damage cause.
- **Fix Hypothesis**: Hierarchical multi-label classifier prioritizing physical damage triggers over financial outcome requests.

### Failure Mode 2: Sarcasm & Implicit Customer Frustration
- **Real Example**: *"Oh fantastic Amazon, 10 days late and still no tracking! Stellar customer service as always!"*
- **Ground Truth**: `ESCALATION_HUMAN_COMPLAINT` (`ESCALATE`).
- **Agent Prediction**: `SHIPPING_DELIVERY_INQUIRY` (`AUTO_HANDLE`).
- **Root Cause & Hypothesis**: Parser detected literal keyword `"tracking"` and missed the sarcastic tone ("stellar service as always").
- **Fix Hypothesis**: Fine-tuned RoBERTa sentiment classifier specialized in implicit sarcasm detection on customer tweets.

### Failure Mode 3: Expired Historical Twitter Short URLs
- **Real Example**: *"Can you re-send the link from tweet #115820? That t.co link gives a 404 error."*
- **Ground Truth**: `FEEDBACK_CHITCHAT_GENERAL` or `ACCOUNT_SECURITY_BILLING`.
- **Agent Prediction**: Drafted reply with a standard `https://amzn.to/help` link.
- **Root Cause & Hypothesis**: Legacy TWCS dataset contains Twitter short links (`t.co`) from 2017 that are now expired.
- **Fix Hypothesis**: Dynamic URL re-mapping layer translating legacy TWCS links to live active Amazon support endpoints.

### Failure Mode 4: Ambiguous Single-Word Customer Messages
- **Real Example**: *"DM sent."* or *"Done."*
- **Ground Truth**: `FEEDBACK_CHITCHAT_GENERAL` (`AUTO_HANDLE`).
- **Agent Prediction**: Low-confidence fallback prediction.
- **Root Cause & Hypothesis**: Ultra-short turns lack semantic context when isolated from prior thread turns.
- **Fix Hypothesis**: Multi-turn conversation context buffer feeding previous turns into classifier.

### Failure Mode 5: Membership Charges Misclassified as Shipping Inquiries
- **Real Example**: *"Why was I charged $14.99 for Prime delivery when I haven't ordered anything?"*
- **Ground Truth**: `ACCOUNT_SECURITY_BILLING` (`ESCALATE`).
- **Agent Prediction**: `SHIPPING_DELIVERY_INQUIRY`.
- **Root Cause & Hypothesis**: Keyword `"delivery"` confused classifier despite query referring to Prime subscription renewal billing.
- **Fix Hypothesis**: Feature weighting discounting `"delivery"` when co-occurring with financial terms like `"$14.99"` or `"charged"`.

---

## 4. "What is Misleading About My Headline Number?" (Mandatory Section)

While achieving **96.5% Intent Accuracy** and **0.9867 Escalation F1** looks impressive, deploying this agent to production based solely on these numbers would be irresponsible for four reasons:

1. **Synthetic Stratification Bias**: The 200 Golden Evaluation Set was constructed via stratified sampling (~25–40 examples per intent). In live Twitter streams, >60% of queries are simple shipping inquiries. Synthetic balance inflates macro F1 performance compared to un-curated live streams.
2. **Dataset Age & E-Commerce Shift**: TWCS was collected in late 2017. Customer query phrasing, Prime Video app UI, and Amazon support URL structures have evolved over the last 9 years.
3. **Single-Turn Context Limitation**: The benchmark evaluates isolated single customer tweets. In reality, support interactions span 3–5 turns; high single-tweet accuracy does not guarantee multi-turn dialog state tracking.
4. **LLM Judge Optimism Bias**: Rubric scoring awards high points (>4.5) to polite replies with working links (`https://amzn.to/help`), even if the reply is somewhat generic.

---

## 5. What We Would Do Next With One More Week

1. **Transformer Classifier Fine-Tuning**: Train a distilled `DeBERTa-v3-small` classifier on 20,000 `@AmazonHelp` pairs to replace TF-IDF/NB, improving implicit sarcasm and compound intent handling.
2. **Multi-Turn Thread Context Buffer**: Extend `agent.py` to track conversation trees across `in_response_to_tweet_id`, maintaining context across turns.
3. **Live Twitter/X API Webhook Integration**: Implement a FastAPI webhook listener connecting to Twitter API v2 for real-time query ingestion and response drafting.
4. **Human-in-the-Loop (HITL) Supervisor Dashboard**: Build a Gradio/React interface enabling human support agents to review, edit, or approve drafted replies and escalation flags before public tweeting.
5. **Dense Vector Search (FAISS + SentenceTransformers)**: Upgrade TF-IDF RAG retrieval to dense embeddings (`bge-small-en-v1.5`) stored in FAISS for semantic precedent retrieval.

---

## 6. Attribution & Integrity Statement (Human vs. AI Assistant Roles)

Per assignment guidelines (*"You may use AI coding assistants freely... Cite anything you borrowed"*):

- **Human Engineer Owned**: System design, problem framing, out-of-scope choices, 7-intent taxonomy, 200-item golden set hand-labeling, triage matrix design, 12 non-obvious engineering decisions, failure mode hypotheses, and headline number critique.
- **AI Assistant Accelerated**: Scikit-Learn pipeline syntax generation, pandas data cleaning script boilerplate, pytest test stub generation, and CLI rich table formatting.
