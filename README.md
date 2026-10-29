# COD RTO Decision Desk

A take-home assignment prototype for a D2C haircare brand. It includes an explainable COD risk model, configurable action thresholds, a WhatsApp flow simulator, and a founder dashboard.

## Quick start (under 10 minutes)

1. Install Python 3.10+.
2. Unzip the repository and open a terminal in this folder.
3. Create/activate a virtual environment (optional but recommended).
4. Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The browser will open the app. The sample dataset is in `data/orders_sample.csv`.

## Decision logic

- Logistic regression estimates RTO risk from historical order/customer fields and simple address quality signals.
- The action thresholds are adjustable in the sidebar; no code changes are needed.
- Low risk: Ship as-is.
- Moderate risk: Ship after confirmation.
- Elevated risk: Nudge to prepaid.
- Very high risk + prior RTO history: Hold/cancel for manual review.
- First-time customers are not automatically cancelled solely because of a high score.

Default thresholds: ship below 20%, confirmation below 36%, prepaid nudge at 36%+, and hold at 50%+ only when prior RTO history exists. These are initial test thresholds, not validated business-optimal values.

## AI / LLM boundary

Deterministic code handles numeric fields, missingness, basic placeholder-name detection, risk thresholds, and all financial calculations. These are repeatable, auditable, and cheaper than an LLM. The Decision Engine includes an optional local LLM address-review button powered by Ollama (default model `qwen2.5:3b`). It can interpret ambiguous Hinglish landmarks and suggest a clarification question. No hosted API key is required, but Ollama and a local model download are required. The LLM must not invent pincode validation, make final financial calculations, or independently cancel a customer. It must not invent pincode validation, make final financial calculations, or independently cancel a customer. Use a structured output schema and human review for uncertain cases.

## Model limitations

- The dataset has 2,018 rows, but only 2,017 have a normalized DELIVERED/RTO outcome; one IN_TRANSIT row is excluded.
- There are 30 duplicate order IDs; duplicates need investigation before production evaluation.
- The dashboard uses 5-fold stratified group out-of-fold probabilities, keeping duplicate order IDs in the same fold. This is a retrospective diagnostic, not time-based validation; repeated-customer leakage may remain. Production evaluation must use a later time period and verify that prior-history fields are point-in-time correct.
- Data may be synthetic or sampled and is not necessarily representative of future traffic.
- The Decision Engine fits on all available labeled rows to score a newly entered order. The historical dashboard uses out-of-fold predictions so each historical row is scored by a fold model that did not train on that row. This does not replace time-based evaluation.
- Dashboard financial impact is an estimate. Run a randomized holdout test before claiming realized savings.

## Business assumptions

Direct RTO cost = ₹65 forward shipping + ₹65 return shipping + ₹25 packaging/handling + (8% × ₹180 damage cost) = ₹169.40 per RTO.

A ₹30 prepaid discount is a starting test, not a proven optimum. The RTO-cost-only saving when a COD order becomes prepaid is (28% − 4%) × ₹169.40 = ₹40.66; after a ₹30 discount this narrow measure is ₹10.66. This excludes changes in conversion and contribution margin, so it must be tested.

## Optional local LLM address review (no API key)

1. Install Ollama for Windows from https://ollama.com/download.
2. Open PowerShell and run `ollama pull qwen2.5:3b` (downloads model weights; size and speed depend on your computer).
3. Ensure Ollama is running, then start this app with `python -m streamlit run app.py`.
4. In the Decision Engine, score an order and click **Review address with local LLM**.

The app calls Ollama at `http://localhost:11434`. The model runs locally; the address text is sent to your local Ollama process, not to a hosted model provider by this code. If Ollama is unavailable, the deterministic address checks still work. The LLM output is untrusted and is presented only as a clarification suggestion; it does not affect the risk score, thresholds, pincode validation, financial calculations, or shipping/cancellation decision. Check the chosen model's license before using it commercially.

If you do not install Ollama, the app remains usable with deterministic address checks, but the LLM capability has not been run.

## WhatsApp simulator

The app previews English/Hinglish templates and simulates YES, NO, PREPAID, COD, unclear replies, and timeout. It sends no real messages or payments. Production must use approved WhatsApp templates, opt-in/consent, secure Shopify checkout links, payment webhooks, and idempotent status updates.

## Files

- `app.py` — Streamlit UI, including scenario estimates for RTO savings, genuine-cancellation contribution loss, and net impact
- `engine.py` — feature engineering, training, action rules, address checks
- `data/orders_sample.csv` — provided sample
- `docs/business_memo.pdf` — founder-facing memo
- `docs/decision_log.md` — key choices and alternatives
- `docs/ai_usage_log.md` — tools, examples of weak AI output, validation
- `COD_RTO_Analysis_Enhanced.xlsx` — companion analysis workbook (provided separately)

## Recommended next steps

1. Confirm duplicate order IDs and status definitions with the data owner.
2. Evaluate on a later time period and inspect calibration by city, courier, new vs returning customer, and acquisition source.
3. Run a randomized holdout / A-B test before automatic cancellation.
4. Add production-grade WhatsApp integration and payment-webhook verification.


### Version 8 note: Streamlit button state

The evaluated order and local LLM response are stored in Streamlit session state. This keeps the review button available across Streamlit reruns and displays the result after the local model call completes.


## Local LLM validation note

The `docs/ai_usage_log.md` records manual smoke tests of the optional local address-review feature, including one misleading city/state response. The feature is intended to suggest clarification questions only. It is not a geocoding, postal validation, or courier-serviceability system.
