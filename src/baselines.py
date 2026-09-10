class TrivialBaselineAgent:
    """
    Baseline 1: Trivial Baseline.
    Predicts majority class intent, always auto-handles, and uses a single static canned reply.
    """
    def process_message(self, customer_text: str) -> dict:
        return {
            "customer_text": customer_text,
            "predicted_intent": "SHIPPING_DELIVERY_INQUIRY",
            "intent_confidence": 1.0,
            "escalation_decision": "AUTO_HANDLE",
            "escalation_reason": "Default trivial baseline rule: auto-handle all incoming messages.",
            "draft_reply": "Thank you for reaching out. Please visit our help page for assistance.",
            "historical_grounding": []
        }

class SimpleBaselineAgent:
    """
    Baseline 2: Simple Baseline.
    Uses simple keyword matching for intent and basic escalation keywords without RAG context.
    """
    def process_message(self, customer_text: str) -> dict:
        t_lower = customer_text.lower()

        # Simple keyword intent
        if "refund" in t_lower or "return" in t_lower:
            intent = "REFUND_RETURN_REQUEST"
        elif "damaged" in t_lower or "broken" in t_lower:
            intent = "DAMAGED_MISSING_ITEM"
        elif "account" in t_lower or "charge" in t_lower:
            intent = "ACCOUNT_SECURITY_BILLING"
        elif "prime" in t_lower or "video" in t_lower:
            intent = "DIGITAL_SERVICES_PRIME"
        elif "deliver" in t_lower or "track" in t_lower or "order" in t_lower:
            intent = "SHIPPING_DELIVERY_INQUIRY"
        elif "terrible" in t_lower or "worst" in t_lower:
            intent = "ESCALATION_HUMAN_COMPLAINT"
        else:
            intent = "FEEDBACK_CHITCHAT_GENERAL"

        # Simple keyword escalation
        if any(w in t_lower for w in ["refund", "damaged", "manager", "terrible", "worst"]):
            decision = "ESCALATE"
            reason = "Simple keyword escalation rule triggered."
        else:
            decision = "AUTO_HANDLE"
            reason = "No simple escalation keyword found."

        reply = f"Hi there! Regarding your query about {intent.replace('_', ' ').lower()}, please reach out or check your account online."

        return {
            "customer_text": customer_text,
            "predicted_intent": intent,
            "intent_confidence": 0.70,
            "escalation_decision": decision,
            "escalation_reason": reason,
            "draft_reply": reply,
            "historical_grounding": []
        }
