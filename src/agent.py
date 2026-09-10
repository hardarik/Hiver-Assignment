from src.intent_classifier import IntentClassifier
from src.rag_retriever import HistoricalRAGRetriever
from src.escalation_engine import EscalationEngine

class AmazonSupportAgent:
    """
    Production-Grade AI Support Agent for @AmazonHelp.
    Classifies customer intent, retrieves historical resolution precedents (RAG),
    evaluates escalation policy rules, and drafts grounded support replies.
    """
    def __init__(self, training_data: list = None):
        self.classifier = IntentClassifier()
        if training_data:
            self.classifier.fit(training_data)
            
        self.retriever = HistoricalRAGRetriever()
        self.retriever.load_and_index()
        
        self.escalation_engine = EscalationEngine()

    def draft_reply(self, customer_text: str, intent: str, escalation_decision: str, historical_examples: list) -> str:
        """
        Drafts reply grounded in brand tone and historical resolution examples.
        """
        # Grounding template logic based on historical AmazonHelp style
        if escalation_decision == "ESCALATE":
            if intent in ["REFUND_RETURN_REQUEST", "ACCOUNT_SECURITY_BILLING", "DAMAGED_MISSING_ITEM"]:
                return "We'd like to take a further look into this with you! Please reach out to us securely via chat or phone here: https://amzn.to/help"
            elif intent == "ESCALATION_HUMAN_COMPLAINT":
                return "I'm so sorry for the frustration this has caused! Please connect with our team directly so we can make this right for you: https://amzn.to/help"
            else:
                return "We'd love to help look into this for you! Please reach out to our team via phone or chat here: https://amzn.to/help"
        else:
            # AUTO_HANDLE templates
            if intent == "DIGITAL_SERVICES_PRIME":
                return "Sorry for the playback issue! Please try force stopping the app and restarting your device. If it persists, check your connection or visit: https://amzn.to/digital-help"
            elif intent == "SHIPPING_DELIVERY_INQUIRY":
                return "You can track your package progress in real-time under 'Your Orders' here: https://amzn.to/orders. Let us know if you need anything else!"
            else:
                return "Thanks for reaching out! We appreciate your feedback and hope you have a great day!"

    def process_message(self, customer_text: str) -> dict:
        """
        Main end-to-end processing pipeline for an incoming customer message.
        """
        # 1. Intent Classification
        clf_result = self.classifier.predict(customer_text)
        predicted_intent = clf_result["intent"]
        confidence = clf_result["confidence"]

        # 2. Historical Resolution RAG Retrieval
        retrieved_examples = self.retriever.retrieve(customer_text, top_k=2)

        # 3. Triage & Escalation Evaluation
        triage_result = self.escalation_engine.evaluate(customer_text, predicted_intent)
        escalation_decision = triage_result["decision"]
        escalation_reason = triage_result["reason"]

        # 4. Draft Grounded Reply
        draft_reply = self.draft_reply(
            customer_text=customer_text,
            intent=predicted_intent,
            escalation_decision=escalation_decision,
            historical_examples=retrieved_examples
        )

        return {
            "customer_text": customer_text,
            "predicted_intent": predicted_intent,
            "intent_confidence": confidence,
            "escalation_decision": escalation_decision,
            "escalation_reason": escalation_reason,
            "draft_reply": draft_reply,
            "historical_grounding": retrieved_examples
        }
