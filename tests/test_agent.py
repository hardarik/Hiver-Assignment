import pytest
from src.agent import AmazonSupportAgent
from src.intent_classifier import IntentClassifier
from src.escalation_engine import EscalationEngine
from src.rag_retriever import HistoricalRAGRetriever

@pytest.fixture
def agent():
    return AmazonSupportAgent()

def test_intent_classifier():
    clf = IntentClassifier()
    res = clf.predict("Where is my package? It is overdue and late!")
    assert res["intent"] == "SHIPPING_DELIVERY_INQUIRY"
    assert res["confidence"] > 0.5

def test_escalation_engine_refund():
    engine = EscalationEngine()
    res = engine.evaluate("I want a refund for my order right now!", "REFUND_RETURN_REQUEST")
    assert res["decision"] == "ESCALATE"
    assert "refund" in res["reason"].lower()

def test_escalation_engine_digital():
    engine = EscalationEngine()
    res = engine.evaluate("My Prime Video app is buffering on Fire TV stick.", "DIGITAL_SERVICES_PRIME")
    assert res["decision"] == "AUTO_HANDLE"
    assert "digital" in res["reason"].lower() or "troubleshooting" in res["reason"].lower()

def test_rag_retriever():
    retriever = HistoricalRAGRetriever()
    retriever.load_and_index()
    results = retriever.retrieve("package not delivered yet", top_k=2)
    assert len(results) == 2
    assert "historical_customer_query" in results[0]
    assert "historical_brand_reply" in results[0]

def test_agent_end_to_end(agent):
    res = agent.process_message("My order arrived damaged with broken glass!")
    assert res["predicted_intent"] == "DAMAGED_MISSING_ITEM"
    assert res["escalation_decision"] == "ESCALATE"
    assert "https://amzn.to/help" in res["draft_reply"]
    assert len(res["historical_grounding"]) > 0
