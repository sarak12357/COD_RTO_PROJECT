# AI Usage Log

## Tools used
- ChatGPT: helped structure the solution, identify a defensible split between LLM and deterministic logic, draft the business memo, and review risks/assumptions.
- Python/pandas/scikit-learn: inspected the supplied workbook, normalized outcome labels, calculated descriptive statistics, and built a baseline logistic-regression prototype.
- Streamlit: used to create a local mock dashboard and interaction simulator.

## Examples of weak AI-style assumptions caught and corrected
1. **Treating every row as a valid training label.** The raw file contains an `IN_TRANSIT` row and inconsistent casing (`Rto`, `delivered`). Corrected by normalizing labels and training only on DELIVERED/RTO outcomes.
2. **Claiming a high-risk city should automatically lose COD.** City/state rates can be noisy, and a hard block would harm genuine customers. Kept geography as a signal in a model, but used a reversible confirmation/prepaid step and human review before cancellation.
3. **Calling avoided shipping cost “net profit.”** A ₹30 discount, payment conversion, order conversion, gross contribution, and genuine cancelled orders all affect net impact. The dashboard labels the result as an estimate and recommends a randomized holdout before claiming realized savings.

## Validation notes
- Data profile and metrics are calculated from the uploaded workbook, not invented.
- Any generated memo figures should be read with the stated limitations: duplicate IDs, small sample, observational data, stratified rather than time-based validation, and assumptions about customer response rates.
- No real WhatsApp messages or payments were sent.
- The optional local LLM address-review extension uses Ollama; it does not call a hosted model provider through this code. The demo still runs without Ollama by using deterministic checks.


## Local LLM address-review option (Version 7)

- Added an optional local Ollama call using `qwen2.5:3b`; no hosted API key is required.
- The LLM receives only the current form's name/address/city/state/pincode fields and returns a structured ambiguity assessment and suggested clarification question.
- The model output is treated as untrusted and validated before display. It cannot change the logistic-regression risk score, validate postal codes, perform financial calculations, or cancel an order.
- The feature requires Ollama to be installed and the model to be downloaded locally. If Ollama is unavailable, deterministic checks remain available.
- This is a prototype integration. Test on representative English/Hindi/Hinglish addresses, compare with human judgments, and disclose that no production address verification or real delivery validation is performed.


## Manual local-LLM validation performed (Version 9 documentation update)

These are informal, manually observed checks on the local Ollama `qwen2.5:3b` model, not a formal benchmark. Outputs can vary by prompt, model version, and run.

| Test input | Observed model behavior | Assessment / limitation |
|---|---|---|
| Address: `Near station` | Flagged the address as ambiguous and asked which station was intended. The deterministic check also flagged the address as very short. | Reasonable clarification suggestion. It does not prove the address is invalid or undeliverable. |
| Address: `Shiv Mandir ke saamne, purani gali, station ke paas` | Recognized the relative landmark wording, flagged the landmark as ambiguous, and asked for a more specific street or landmark. | Useful Hinglish interpretation, but it may over-flag locally understandable addresses. A human/operator should decide whether clarification is needed. |
| Address: `Flat 12, MG Road, near City Mall` | Did not identify a clear ambiguity and reported no specific issue. | A “no ambiguity found” result is not address verification. It does not establish that the street, landmark, pincode, or delivery coverage is valid. |
| Inconsistent city/state example: city entered as `Pune`, state entered as `Assam` | The model produced a misleading explanation, saying Assam is not a city and asking for a city within Assam. | Failure/caveat: the model should not be relied on to validate city–state consistency. Deterministic validation against a trusted reference dataset and human review are needed. |

### Guardrails retained
- LLM output is advisory only. It must not change the RTO probability, action thresholds, financial calculations, pincode validation, or cancel/hold decision.
- The test set is tiny and manually selected; it does not establish accuracy, fairness, or production readiness.
- Before deployment, evaluate a representative, consented sample of English/Hindi/Hinglish addresses against human-labelled judgments, measure false-positive and false-negative rates, and keep an operator override.
