import re

class EscalationEngine:
    """
    Triage and Escalation Decision Engine.
    Evaluates incoming customer queries against Intent Policies, Actionability Risk,
    Customer Anger / Sentiment, and Security Privacy Guardrails.
    """
    def __init__(self):
        # Intents that MANDATE escalation due to security, loss claims, or transactional account access
        self.mandatory_escalate_intents = {
            "REFUND_RETURN_REQUEST",
            "ACCOUNT_SECURITY_BILLING",
            "DAMAGED_MISSING_ITEM",
            "ESCALATION_HUMAN_COMPLAINT"
        }

        # Profanity & Anger escalation triggers
        self.anger_keywords = [
            "pissed", "furious", "terrible", "worst", "sucks", "lawsuit", 
            "lawyer", "bbb", "scam", "useless", "lie", "lied", "horrible",
            "supervisor", "manager", "disappointed", "unacceptable"
        ]

    def evaluate(self, customer_text: str, intent: str) -> dict:
        """
        Evaluate customer query and classified intent to return escalation decision
        and explicit human-readable reason.
        """
        t_lower = customer_text.lower()

        # 1. Customer Anger / Escalation Request Check
        anger_matches = [w for w in self.anger_keywords if w in t_lower]
        if anger_matches:
            return {
                "decision": "ESCALATE",
                "reason": f"High customer frustration/escalation signal detected (keywords: {', '.join(anger_matches[:2])}). Requires human supervisor empathy.",
                "risk_score": 0.90
            }

        # 2. Mandatory Policy Intent Check
        if intent in self.mandatory_escalate_intents:
            reasons = {
                "REFUND_RETURN_REQUEST": "Financial order refund / return processing requires account lookup and secure agent transaction approval.",
                "ACCOUNT_SECURITY_BILLING": "Involves sensitive account credentials, billing details, or login access requiring authenticated support portal.",
                "DAMAGED_MISSING_ITEM": "Physical damage or lost item claims require customer verification and formal return shipping authorization.",
                "ESCALATION_HUMAN_COMPLAINT": "Severe satisfaction dispute requiring human customer service specialist intervention."
            }
            return {
                "decision": "ESCALATE",
                "reason": reasons.get(intent, "Policy mandates human agent handling for account-level operations."),
                "risk_score": 0.85
            }

        # 3. Contextual Shipping & Delivery Check
        if intent == "SHIPPING_DELIVERY_INQUIRY":
            # Simple tracking info vs lost package dispute
            if any(k in t_lower for k in ["not delivered", "days late", "overdue", "stolen", "wrong address"]):
                return {
                    "decision": "ESCALATE",
                    "reason": "Overdue or disputed shipping delivery requires human carrier tracking investigation.",
                    "risk_score": 0.75
                }
            return {
                "decision": "AUTO_HANDLE",
                "reason": "Standard package tracking inquiry can be self-resolved via Amazon tracking portal.",
                "risk_score": 0.20
            }

        # 4. Digital Services / Prime Troubleshooting
        if intent == "DIGITAL_SERVICES_PRIME":
            return {
                "decision": "AUTO_HANDLE",
                "reason": "Digital service / Prime playback troubleshooting can be self-resolved via standard device app reboot guidance.",
                "risk_score": 0.15
            }

        # 5. General Feedback / Chitchat
        return {
            "decision": "AUTO_HANDLE",
            "reason": "General brand comment or compliment suitable for automated polite response.",
            "risk_score": 0.10
        }
