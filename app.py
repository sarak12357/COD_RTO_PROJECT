import streamlit as st
import pandas as pd
import requests
from engine import add_features, train_model, historical_oof_predictions, choose_action, address_review, CAT_COLS, NUM_COLS
from llm_address_review import review_address_with_local_llm

st.set_page_config(page_title="COD RTO Decision Desk", page_icon="📦", layout="wide")
st.title("📦 COD RTO Decision Desk")
st.caption("Take-home assignment demo • Uses historical sample data • All financial outputs are estimates")

with st.sidebar:
    st.header("Decision thresholds")
    ship_max = st.slider("Ship as-is below risk", 0.05, 0.45, 0.20, 0.01)
    confirm_max = st.slider("Confirmation below risk", ship_max + 0.01, 0.60, max(0.36, ship_max + 0.01), 0.01)
    cancel_min = st.slider("Hold threshold", confirm_max + 0.01, 0.90, max(0.50, confirm_max + 0.01), 0.01)
    st.caption("Hold/cancel is only triggered when risk is high AND customer has prior RTO history. New customers are protected from automatic cancellation.")
    discount = st.number_input("Prepaid discount (₹)", min_value=0, max_value=100, value=30, step=5)
    confirmation_hours = st.selectbox("Confirmation timeout (hours)", [6, 12, 24, 36], index=2)

@st.cache_resource
def load_model():
    model, training_df = train_model()
    return model, training_df
try:
    model, train_df = load_model()
except Exception as e:
    st.error(f"Could not load the sample dataset/model: {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Decision engine", "WhatsApp simulator", "Founder dashboard"])
with tab1:
    st.subheader("Score a new COD order")
    c1, c2, c3 = st.columns(3)
    with c1:
        name = st.text_input("Customer name", "Aarav Sharma")
        phone = st.text_input("Phone", "9876543210")
        address = st.text_area("Delivery address", "Flat 12, MG Road, near City Mall")
        city = st.text_input("City", "Pune")
        state = st.selectbox("State", sorted(train_df["state"].dropna().astype(str).unique()), index=0)
    with c2:
        pincode = st.text_input("Pincode", "411001")
        order_value = st.number_input("Order value (₹)", 100, 10000, 649, 50)
        num_items = st.number_input("Number of items", 1, 20, 1)
        prior_orders = st.number_input("Previous orders", 0, 100, 0)
        prior_rto = st.number_input("Previous RTOs", 0, 20, 0)
    with c3:
        courier = st.selectbox("Courier", sorted(train_df["courier"].dropna().astype(str).unique()))
        utm_source = st.selectbox("Acquisition source", sorted(train_df["utm_source"].dropna().astype(str).unique()))
        st.info("This demo predicts risk from historical patterns and basic data-quality signals. It is not a guarantee of delivery.")

    if st.button("Evaluate order", type="primary"):
        row = pd.DataFrame([{
            "customer_name":name, "phone":phone, "address":address, "city":city, "state":state,
            "pincode":pincode, "order_value":order_value, "num_items":num_items,
            "customer_prior_orders":prior_orders, "customer_prior_rto":prior_rto,
            "courier":courier, "utm_source":utm_source
        }])
        row = add_features(row)
        probability = float(model.predict_proba(row[CAT_COLS + NUM_COLS])[0,1])
        action, reason = choose_action(probability, prior_rto, {"ship_max":ship_max, "confirm_max":confirm_max, "cancel_min":cancel_min})
        # Persist evaluated results across Streamlit reruns so the LLM button can work independently.
        st.session_state["cod_last_order_record"] = row.iloc[0].to_dict()
        st.session_state["cod_last_probability"] = probability
        st.session_state["cod_last_action"] = action
        st.session_state["cod_last_reason"] = reason
        st.session_state["cod_last_llm_result"] = None
        st.session_state["cod_last_llm_error"] = None

    order_record = st.session_state.get("cod_last_order_record")
    if order_record is not None:
        probability = st.session_state["cod_last_probability"]
        action = st.session_state["cod_last_action"]
        reason = st.session_state["cod_last_reason"]
        st.metric("Predicted RTO risk", f"{probability:.1%}")
        st.subheader(f"Recommended action: {action}")
        st.write(reason)
        flags = address_review(order_record)
        if flags:
            st.warning("Deterministic address/data-quality checks")
            for flag in flags:
                st.write(f"• {flag}")
        else:
            st.success("No basic address-quality flags found. This does not prove the address is valid.")

        st.markdown("#### Optional local LLM address review")
        st.caption("Runs on your own computer through Ollama. No hosted API key is required. The LLM can suggest clarification only; it cannot change the risk score or cancel an order.")
        if st.button("Review address with local LLM"):
            try:
                with st.spinner("Reviewing address locally…"):
                    st.session_state["cod_last_llm_result"] = review_address_with_local_llm(order_record)
                st.session_state["cod_last_llm_error"] = None
            except requests.exceptions.ConnectionError:
                st.session_state["cod_last_llm_result"] = None
                st.session_state["cod_last_llm_error"] = "Ollama is not reachable. Check that Ollama is running and qwen2.5:3b is installed."
            except Exception as e:
                st.session_state["cod_last_llm_result"] = None
                st.session_state["cod_last_llm_error"] = f"Local LLM review failed: {e}"

        llm_error = st.session_state.get("cod_last_llm_error")
        llm_result = st.session_state.get("cod_last_llm_result")
        if llm_error:
            st.error(llm_error)
        if llm_result:
            if llm_result["address_ambiguous"]:
                st.warning("The local LLM thinks clarification may be needed.")
            else:
                st.success("The local LLM did not identify a clear ambiguity. This is not proof the address is deliverable.")
            st.write("**Issues**")
            if llm_result["issues"]:
                for issue in llm_result["issues"]:
                    st.write(f"• {issue}")
            else:
                st.write("No specific ambiguity reported.")
            if llm_result["clarification_question"]:
                st.write("**Suggested customer question:** " + llm_result["clarification_question"])
            if llm_result["language_understanding"]:
                st.write("**Hinglish interpretation:** " + llm_result["language_understanding"])
            st.caption(f"Local model confidence: {llm_result['confidence']}. Treat this as a suggestion for human review.")
        st.caption("LLM boundary: numeric risk, thresholds, pincode validation, and financial calculations remain deterministic.")
with tab2:
    st.subheader("Mock WhatsApp flow")
    action_type = st.radio("Choose flow", ["Ship after confirmation", "Nudge to prepaid"], horizontal=True)
    language = st.radio("Message language", ["Hinglish", "English"], horizontal=True)
    if action_type == "Ship after confirmation":
        msg = ("Namaste! Aapka haircare order ₹649 ka place hua hai. Dispatch karne ke liye YES reply karein. "
               "Agar order nahi chahiye, NO reply karein. 24 ghante mein reply na aane par order hold hoga.") if language=="Hinglish" else (
               "Hi! Your haircare order worth ₹649 is ready to process. Reply YES to confirm dispatch or NO to cancel. "
               "If we don't hear back within 24 hours, the order will be placed on hold.")
    else:
        msg = (f"Namaste! Online payment par ₹{discount} off milega. Abhi prepaid karein aur order ₹{649-discount} mein confirm karein. "
               "COD continue karna ho to COD reply karein.") if language=="Hinglish" else (
               f"Pay online and get ₹{discount} off your ₹649 order. Your prepaid total will be ₹{649-discount}. "
               "Reply PREPAID to choose online payment or COD to keep cash on delivery.")
    st.text_area("Message preview", msg, height=100)
    reply = st.selectbox("Simulate customer reply", ["YES", "NO", "PREPAID", "COD", "Maybe later", "No reply (24h timeout)"])
    if st.button("Process reply"):
        r = reply.strip().upper()
        if r == "YES": result = "Confirmed. Release order for dispatch."
        elif r == "NO": result = "Customer declined. Cancel and log reason."
        elif r == "PREPAID": result = "Send secure Shopify payment link. Release only after payment webhook confirms success."
        elif r == "COD": result = "Customer keeps COD. Route to confirmation policy or manual review; do not pretend payment changed."
        elif r == "NO REPLY (24H TIMEOUT)": result = f"No reply after {confirmation_hours} hours. Hold order, send one final reminder, then cancel only under configured policy. Notify customer."
        else: result = "Unrecognized reply. Send quick-reply options YES / NO (or PREPAID / COD), allow one clarification, then route to human review."
        st.success(result)
    st.caption("Mock only: no real WhatsApp messages or payments are sent. In production, use approved WhatsApp templates, consent, idempotent webhooks, and a verified Shopify payment status.")

with tab3:
    st.subheader("Founder dashboard")
    # Use out-of-fold predictions for retrospective views; keep the full-data model
    # above only for scoring newly entered orders in the Decision Engine tab.
    try:
        features = historical_oof_predictions("data/orders_sample.csv")
    except Exception as e:
        st.error(f"Could not generate historical out-of-fold predictions: {e}")
        st.stop()
    # Use the OOF probabilities returned with the historical rows (not an undefined
    # variable and not the full-data model's in-sample predictions).
    probs = features["risk_probability"].to_numpy()
    features["action"] = [choose_action(p,r,{"ship_max":ship_max,"confirm_max":confirm_max,"cancel_min":cancel_min})[0]
                          for p,r in zip(probs,features["customer_prior_rto"])]
    dates = pd.to_datetime(features["order_datetime"], errors="coerce")
    features["date"] = dates.dt.date
    selected_date = st.date_input("Dashboard date", value=dates.max().date())
    day = features[features["date"] == selected_date]
    cols = st.columns(4)
    for col, action in zip(cols, ["Ship as-is","Ship after confirmation","Nudge to prepaid","Hold / cancel"]):
        col.metric(action, int((day["action"]==action).sum()))
    st.write("Action counts for selected date")
    st.bar_chart(day["action"].value_counts())
    st.markdown("#### Estimated financial impact for selected date")
    rto_cost = 65 + 65 + 25 + 0.08*180
    delivered_contribution = 649 * 0.62
    cod_rto_rate = 0.28
    prepaid_rto_rate = 0.04
    prepaid_acceptance = 0.50
    confirmation_response = 0.60
    confirmation_rto_reduction = 0.10
    # These are explicit pilot assumptions, not measured treatment effects.
    nudge_orders = day[day["action"] == "Nudge to prepaid"]
    confirm_orders = day[day["action"] == "Ship after confirmation"]
    cancel_orders = day[day["action"] == "Hold / cancel"]
    nudge_conversions = len(nudge_orders) * prepaid_acceptance
    confirm_successes = len(confirm_orders) * confirmation_response
    cancel_rto_avoided = float(cancel_orders["risk_probability"].sum())
    nudge_rto_avoided = nudge_conversions * (cod_rto_rate - prepaid_rto_rate)
    confirm_rto_avoided = confirm_successes * confirmation_rto_reduction
    estimated_rtos_avoided = nudge_rto_avoided + confirm_rto_avoided + cancel_rto_avoided
    rto_savings = estimated_rtos_avoided * rto_cost
    discount_cost = nudge_conversions * discount
    cancelled_genuine_orders = float((1 - cancel_orders["risk_probability"]).sum())
    lost_contribution = cancelled_genuine_orders * delivered_contribution
    net_impact = rto_savings - discount_cost - lost_contribution

    k1, k2, k3 = st.columns(3)
    k1.metric("Estimated RTO costs avoided", f"₹{rto_savings:,.2f}")
    k2.metric("Estimated contribution lost on genuine cancellations", f"₹{lost_contribution:,.2f}")
    k3.metric("Estimated net ₹ impact", f"₹{net_impact:,.2f}", help="RTO cost savings minus expected prepaid discounts and estimated contribution lost on cancelled genuine orders.")
    st.write(f"**Direct cost per RTO:** ₹{rto_cost:.2f} = ₹65 forward + ₹65 return + ₹25 handling + (8% × ₹180 damage cost).")
    st.write(f"**Estimated genuine cancellations:** {cancelled_genuine_orders:.2f} orders, weighted by (1 − predicted RTO probability). **Contribution per delivered order:** ₹{delivered_contribution:.2f} (₹649 × 62%).")
    st.write(f"**Prepaid assumption:** {prepaid_acceptance:.0%} of nudged orders convert; each conversion is assumed to reduce RTO probability from 28% to 4%. Discount cost is estimated as ₹{discount_cost:,.2f}.")
    st.write(f"**Confirmation assumption:** {confirmation_response:.0%} respond and confirmation reduces RTO probability by {confirmation_rto_reduction:.0%} (10 percentage points) for those orders. Cancelled orders avoid expected RTO costs in proportion to their predicted RTO probabilities.")
    st.warning("These are scenario estimates, not realized or causal savings. They depend on assumed response rates and risk probabilities. Validate with a randomized holdout, actual prepaid payment events, cancellation reviews, and contribution-margin accounting before using for live decisions.")
    st.markdown("#### Model evaluation")
    st.info("Historical risk scores use 5-fold stratified group out-of-fold predictions; duplicate order IDs are kept in the same fold. This reduces direct train/test overlap for duplicate orders, but it is not time-based validation and does not prevent repeat-customer leakage. Treat it as a diagnostic, not proof of future performance.")
    st.write("The sample contains repeated order IDs and one IN_TRANSIT row. Confirm duplicate semantics and verify that customer-prior fields were computed only from information available before each order. Before production, evaluate on a later time period and monitor calibration, subgroup performance, and false cancellations.")
