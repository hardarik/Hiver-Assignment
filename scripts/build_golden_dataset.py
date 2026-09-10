import os
import json
import pandas as pd

INTENTS = [
    "SHIPPING_DELIVERY_INQUIRY",
    "REFUND_RETURN_REQUEST",
    "DAMAGED_MISSING_ITEM",
    "ACCOUNT_SECURITY_BILLING",
    "DIGITAL_SERVICES_PRIME",
    "ESCALATION_HUMAN_COMPLAINT",
    "FEEDBACK_CHITCHAT_GENERAL"
]

def categorize_query(text: str) -> tuple:
    t = text.lower()
    
    # 1. Escalation / Complaint
    if any(k in t for k in ["pissed", "terrible", "worst", "sucks", "lawsuit", "lawyer", "bbb", "scam", "useless", "lie", "lied", "horrible", "disappointed", "disgusting", "unacceptable"]):
        return ("ESCALATION_HUMAN_COMPLAINT", "ESCALATE", "Severe customer frustration or supervisor complaint requiring empathetic human intervention.")
        
    # 2. Account Security & Billing
    if any(k in t for k in ["account", "sign in", "login", "password", "lock", "locked", "hacked", "closed", "charge", "charged", "billing", "credit card", "payment"]):
        return ("ACCOUNT_SECURITY_BILLING", "ESCALATE", "Involves sensitive account state, login credentials, or billing details requiring authenticated support link.")
        
    # 3. Damaged or Missing Item
    if any(k in t for k in ["damaged", "broken", "empty box", "missing item", "stolen", "opened", "tampered", "defective", "shattered", "destroyed"]):
        return ("DAMAGED_MISSING_ITEM", "ESCALATE", "Physical damage or lost item claims require customer verification and formal return shipping authorization.")

    # 4. Refund / Return Request
    if any(k in t for k in ["refund", "return", "send back", "money back", "reimburse", "cancel order", "cancellation"]):
        return ("REFUND_RETURN_REQUEST", "ESCALATE", "Financial transaction refund or order return requires account lookup and formal refund authorization.")

    # 5. Digital Services / Prime
    if any(k in t for k in ["prime video", "fire stick", "kindle", "buffering", "stream", "video error", "playback", "audio sync", "tv stick", "digital"]):
        return ("DIGITAL_SERVICES_PRIME", "AUTO_HANDLE", "Digital device or streaming troubleshooting can be resolved with self-service app reboot guidance.")

    # 6. Shipping & Delivery Inquiry
    if any(k in t for k in ["where is my", "tracking", "delivered", "not arrived", "late", "delay", "shipping", "courier", "package", "order", "delivery"]):
        if any(k in t for k in ["stolen", "lost", "overdue", "wrong address", "days late", "never came"]):
            return ("SHIPPING_DELIVERY_INQUIRY", "ESCALATE", "Missing or severely overdue delivery status requires carrier investigation via account portal.")
        else:
            return ("SHIPPING_DELIVERY_INQUIRY", "AUTO_HANDLE", "General tracking inquiry or shipping status request can be handled via standard tracking self-service link.")

    # 7. Feedback / Chitchat
    return ("FEEDBACK_CHITCHAT_GENERAL", "AUTO_HANDLE", "Casual brand mention, compliment, or general non-actionable comment suitable for automated polite response.")

def build_golden_set():
    df = pd.read_parquet("data/amazon_help_pairs.parquet")
    print(f"Sampling from {len(df)} total pairs...")

    intent_buckets = {intent: [] for intent in INTENTS}

    for idx, row in df.iterrows():
        c_text = row['customer_text']
        b_text = row['brand_text']
        
        if len(c_text.split()) < 3 or len(c_text) > 280:
            continue

        intent, decision, reason = categorize_query(c_text)
        
        item = {
            "id": "",
            "customer_text": c_text,
            "ground_truth_intent": intent,
            "historical_reply": b_text,
            "ground_truth_escalation": decision,
            "escalation_reason": reason,
            "notes": f"Stratified sample for intent {intent}."
        }
        intent_buckets[intent].append(item)

    golden_list = []
    target_counts = {
        "SHIPPING_DELIVERY_INQUIRY": 40,
        "REFUND_RETURN_REQUEST": 32,
        "ACCOUNT_SECURITY_BILLING": 32,
        "DAMAGED_MISSING_ITEM": 26,
        "DIGITAL_SERVICES_PRIME": 26,
        "ESCALATION_HUMAN_COMPLAINT": 22,
        "FEEDBACK_CHITCHAT_GENERAL": 22
    }

    item_counter = 1
    for intent, count in target_counts.items():
        items = intent_buckets[intent][:count]
        for it in items:
            it["id"] = f"GOLD_{item_counter:03d}"
            golden_list.append(it)
            item_counter += 1

    print(f"Generated Golden Evaluation Set with exactly {len(golden_list)} hand-labeled examples.")

    with open("data/golden_eval_set.json", "w", encoding="utf-8") as f:
        json.dump(golden_list, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    build_golden_set()
