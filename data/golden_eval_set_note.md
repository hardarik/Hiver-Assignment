# Golden Evaluation Set Methodology & Documentation

## Overview
The Golden Evaluation Set consists of **200 hand-curated customer support examples** derived from real `@AmazonHelp` Twitter customer interactions. This dataset serves as the benchmark for evaluating intent classification, escalation triage accuracy, and historical reply grounding.

## Intent Breakdown (200 Total Examples)
| Intent Category | Count | % of Golden Set | Escalation Default |
|---|---|---|---|
| `SHIPPING_DELIVERY_INQUIRY` | 38 | 19.0% | AUTO_HANDLE / ESCALATE (Contextual) |
| `REFUND_RETURN_REQUEST` | 32 | 16.0% | ESCALATE |
| `ACCOUNT_SECURITY_BILLING` | 30 | 15.0% | ESCALATE |
| `DAMAGED_MISSING_ITEM` | 26 | 13.0% | ESCALATE |
| `DIGITAL_SERVICES_PRIME` | 26 | 13.0% | AUTO_HANDLE |
| `ESCALATION_HUMAN_COMPLAINT` | 24 | 12.0% | ESCALATE |
| `FEEDBACK_CHITCHAT_GENERAL` | 24 | 12.0% | AUTO_HANDLE |

## Sampling Strategy
1. **Stratified Sampling**: Selected to ensure representation across all 7 target intents rather than mirroring the raw class distribution (which is heavily biased toward shipping queries).
2. **Length & Quality Filtering**: Excluded raw URL-only tweets, foreign language tweets, and ultra-short fragments (<4 words).
3. **Edge Case Injection**: Included boundary examples such as frustrated shipping queries (testing escalation decision threshold), polite refund requests, and casual feedback.

## Labeling Guidelines
- **`ground_truth_intent`**: Single primary intent assigned based on the customer's main objective.
- **`ground_truth_escalation`**:
  - `ESCALATE`: Assigned whenever account authentication, financial refund execution, damaged goods claim inspection, or human supervisor empathy is required.
  - `AUTO_HANDLE`: Assigned when standard public policy links, troubleshooting steps, or polite acknowledgment suffice without account state changes.
- **`escalation_reason`**: Explicit policy justification explaining why the query requires human intervention vs auto-resolution.
