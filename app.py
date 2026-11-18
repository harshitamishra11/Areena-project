import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import datetime

# Page Config
st.set_page_config(page_title="UPI-Shield | Fraud Analytics", layout="wide")
st.title("🛡️ UPI-Shield: Digital Payment Fraud Detection System")
st.markdown("**PS-17 Solution** | Hybrid Behavioral & ML Fraud Analytics on Synthetic UPI Logs")

# 1. Generate Synthetic UPI Dataset
@st.cache_data
def generate_synthetic_data(n=500):
    np.random.seed(42)
    users = [f"user_{i}@okaxis" for i in range(1, 51)]
    merchants = [f"merchant_{i}@okicici" for i in range(1, 30)]
    mules = ["lottery_claim@ybl", "kyc_update_help@paytm", "refund_desk99@oksbi"]
    
    data = []
    base_time = datetime.datetime(2026, 11, 18, 8, 0, 0)
    
    for i in range(n):
        sender = np.random.choice(users)
        receiver = np.random.choice(merchants + users)
        amount = round(np.random.exponential(scale=1200) + 50, 2)
        tx_time = base_time + datetime.timedelta(minutes=np.random.randint(0, 900))
        device_changed = 0
        tx_velocity_5m = np.random.choice([1, 1, 1, 2], p=[0.7, 0.15, 0.1, 0.05])
        is_new_beneficiary = np.random.choice([0, 1], p=[0.8, 0.2])
        tx_type = np.random.choice(["P2M", "P2P", "COLLECT"], p=[0.6, 0.35, 0.05])
        
        # Inject Fraud Scenarios (~10% of records)
        fraud_scenario = "Normal"
        if i % 10 == 0:
            scenario = np.random.choice(["ATO", "Social_Eng", "Mule_Activity", "Velocity_Burst"])
            if scenario == "ATO":
                device_changed = 1
                amount = round(np.random.uniform(25000, 95000), 2)
                is_new_beneficiary = 1
                fraud_scenario = "Account Takeover (New Device + High Value)"
            elif scenario == "Social_Eng":
                tx_type = "COLLECT"
                receiver = np.random.choice(mules)
                amount = round(np.random.uniform(15000, 49999), 2)
                tx_time = tx_time.replace(hour=np.random.choice([1, 2, 3, 23]))
                fraud_scenario = "Social Engineering (Suspicious Collect Request)"
            elif scenario == "Mule_Activity":
                receiver = np.random.choice(mules)
                is_new_beneficiary = 1
                amount = round(np.random.uniform(10000, 45000), 2)
                fraud_scenario = "Suspicious Beneficiary (Known Mule Pattern)"
            elif scenario == "Velocity_Burst":
                tx_velocity_5m = np.random.randint(6, 15)
                amount = round(np.random.uniform(9000, 19000), 2)
                fraud_scenario = "Abnormal Sequence (Rapid Velocity Burst)"

        data.append({
            "TXN_ID": f"UPI{100000+i}",
            "Timestamp": tx_time,
            "Hour": tx_time.hour,
            "Sender_VPA": sender,
            "Receiver_VPA": receiver,
            "Amount_INR": amount,
            "Type": tx_type,
            "Device_Changed": device_changed,
            "Tx_Velocity_5Min": tx_velocity_5m,
            "New_Beneficiary": is_new_beneficiary,
            "Injected_Pattern": fraud_scenario
        })
    return pd.DataFrame(data)

df = generate_synthetic_data()

# 2. Hybrid Fraud Scoring Engine (ML + Domain Rules)
features = df[["Amount_INR", "Hour", "Device_Changed", "Tx_Velocity_5Min", "New_Beneficiary"]]
iso_forest = IsolationForest(contamination=0.12, random_state=42)
df["ML_Anomaly"] = iso_forest.fit_predict(features) # -1 is anomaly

def calculate_risk(row):
    score = 0
    reasons = []
    if row["Device_Changed"] == 1 and row["Amount_INR"] > 10000:
        score += 45
        reasons.append("New Device + High Amount (ATO Risk)")
    if row["Tx_Velocity_5Min"] >= 5:
        score += 40
        reasons.append(f"High Velocity ({row['Tx_Velocity_5Min']} txns/5m)")
    if row["Type"] == "COLLECT" and row["Amount_INR"] > 5000:
        score += 35
        reasons.append("High-Value Collect Request (Social Eng)")
    if row["New_Beneficiary"] == 1 and row["Amount_INR"] > 15000:
        score += 25
        reasons.append("Large Transfer to New Beneficiary")
    if row["Hour"] in [0, 1, 2, 3, 4]:
        score += 15
        reasons.append("Odd-Hour Activity")
    if row["ML_Anomaly"] == -1:
        score += 20
        reasons.append("ML Behavioral Outlier")
        
    score = min(score, 100)
    status = "🚨 HIGH RISK" if score >= 60 else ("⚠️ REVIEW" if score >= 35 else "✅ SAFE")
    return pd.Series([score, status, ", ".join(reasons) if reasons else "Normal Behavior"])

df[["Risk_Score", "Status", "Flag_Reasons"]] = df.apply(calculate_risk, axis=1)

# 3. Dashboard UI
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Synthetic Txns", len(df))
col2.metric("High Risk Flagged", len(df[df["Status"] == "🚨 HIGH RISK"]))
col3.metric("Under Manual Review", len(df[df["Status"] == "⚠️ REVIEW"]))
col4.metric("Volume Protected (₹)", f"₹{df[df['Risk_Score'] >= 60]['Amount_INR'].sum():,.0f}")

st.subheader("🔍 Flagged Suspicious Transactions (Actionable Analyst Queue)")
filter_status = st.multiselect("Filter by Risk Status:", ["🚨 HIGH RISK", "⚠️ REVIEW", "✅ SAFE"], default=["🚨 HIGH RISK", "⚠️ REVIEW"])
filtered_df = df[df["Status"].isin(filter_status)].sort_values(by="Risk_Score", ascending=False)
st.dataframe(filtered_df[["TXN_ID", "Timestamp", "Sender_VPA", "Receiver_VPA", "Amount_INR", "Type", "Risk_Score", "Status", "Flag_Reasons"]], use_container_width=True)

st.subheader("📊 Fraud Pattern Breakdown")
st.bar_chart(df[df["Risk_Score"] >= 35]["Injected_Pattern"].value_counts())
