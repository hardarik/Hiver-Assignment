import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from src.agent import AmazonSupportAgent
from src.baselines import TrivialBaselineAgent, SimpleBaselineAgent
from src.llm_judge import LLMJudgeEvaluator, validate_judge_human_agreement

def run_evaluation(golden_path: str = "data/golden_eval_set.json") -> dict:
    """
    Evaluates Proposed Agent, Trivial Baseline, and Simple Baseline on Golden Dataset.
    """
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden dataset file not found at {golden_path}")

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    print(f"Loaded {len(golden_set)} golden evaluation examples.")

    # Initialize Agents
    proposed_agent = AmazonSupportAgent(training_data=golden_set)
    trivial_agent = TrivialBaselineAgent()
    simple_agent = SimpleBaselineAgent()
    judge = LLMJudgeEvaluator()

    agents = {
        "Trivial Baseline": trivial_agent,
        "Simple Baseline": simple_agent,
        "Proposed System": proposed_agent
    }

    results = {}

    for name, agent in agents.items():
        print(f"\nEvaluating {name}...")
        y_true_intent = []
        y_pred_intent = []
        y_true_escalate = []
        y_pred_escalate = []
        reply_scores = []

        for item in golden_set:
            c_text = item["customer_text"]
            gold_intent = item["ground_truth_intent"]
            gold_escalate = item["ground_truth_escalation"]
            hist_reply = item["historical_reply"]

            res = agent.process_message(c_text)

            y_true_intent.append(gold_intent)
            y_pred_intent.append(res["predicted_intent"])

            y_true_escalate.append(gold_escalate)
            y_pred_escalate.append(res["escalation_decision"])

            # Reply Quality via LLM Judge
            scores = judge.evaluate_reply(
                customer_text=c_text,
                predicted_intent=res["predicted_intent"],
                escalation_decision=res["escalation_decision"],
                draft_reply=res["draft_reply"],
                historical_reply=hist_reply
            )
            reply_scores.append(scores["overall_score"])

        # Calculate Intent Metrics
        intent_acc = accuracy_score(y_true_intent, y_pred_intent)
        intent_f1_macro = f1_score(y_true_intent, y_pred_intent, average='macro', zero_division=0)
        intent_f1_weighted = f1_score(y_true_intent, y_pred_intent, average='weighted', zero_division=0)

        # Calculate Escalation Metrics (Focusing on ESCALATE class)
        esc_p = precision_score(y_true_escalate, y_pred_escalate, pos_label="ESCALATE", zero_division=0)
        esc_r = recall_score(y_true_escalate, y_pred_escalate, pos_label="ESCALATE", zero_division=0)
        esc_f1 = f1_score(y_true_escalate, y_pred_escalate, pos_label="ESCALATE", zero_division=0)
        cm = confusion_matrix(y_true_escalate, y_pred_escalate, labels=["AUTO_HANDLE", "ESCALATE"])

        avg_reply_quality = round(float(np.mean(reply_scores)), 2)

        results[name] = {
            "intent_accuracy": round(float(intent_acc), 4),
            "intent_f1_macro": round(float(intent_f1_macro), 4),
            "intent_f1_weighted": round(float(intent_f1_weighted), 4),
            "escalation_precision": round(float(esc_p), 4),
            "escalation_recall": round(float(esc_r), 4),
            "escalation_f1": round(float(esc_f1), 4),
            "confusion_matrix": cm.tolist(),
            "avg_reply_quality_score": avg_reply_quality
        }

    # Compute Judge vs Human Agreement
    agreement_metrics = validate_judge_human_agreement(golden_set)
    results["judge_human_agreement"] = agreement_metrics

    return results

if __name__ == "__main__":
    res = run_evaluation()
    print("\n--- FINAL BENCHMARK EVALUATION RESULTS ---")
    print(json.dumps(res, indent=2))
