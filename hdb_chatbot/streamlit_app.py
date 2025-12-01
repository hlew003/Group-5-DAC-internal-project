import os

import pandas as pd
import streamlit as st
from openai import OpenAI

from ml_stub import predict_resale_price  # later swap this to your real ML pipeline


# ---------- LOAD API KEY (NO python-dotenv) ----------

def load_api_key() -> str:
    """
    Load OPENAI_API_KEY from environment variable or from a .env file manually.
    """
    # 1) Try environment variable first
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    # 2) Fallback: read from .env in current folder
    try:
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass

    # 3) Still nothing -> clear error
    raise RuntimeError(
        "OPENAI_API_KEY not found. Set it in your shell (export OPENAI_API_KEY=...) "
        "or in a .env file in the same folder as this script."
    )


api_key = load_api_key()
client = OpenAI(api_key=api_key)
OPENAI_MODEL = "gpt-4.1-mini"  # cheaper model for demo


# ---------- OPENAI CHAT HELPER ----------

def ask_openai(messages):
    """
    messages is a list of dicts: [{"role": "...", "content": "..."}]
    Returns assistant reply text.
    """
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.4,
    )
    return response.choices[0].message.content


# ---------- STREAMLIT PAGE SETUP ----------

st.set_page_config(
    page_title="HDB Resale Chatbot",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 HDB Resale Chatbot & Valuation Demo")
st.caption("OpenAI-powered chatbot + ML pipeline hook for HDB resale valuations.")


# =====================================================================
#  SECTION 1: CHAT
# =====================================================================

st.header("💬 Chat about HDB Resale")

col_chat, col_val = st.columns([2, 1])  # chat wider, forms narrower

with col_chat:
    # Initialise chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that answers questions about HDB resale flats "
                    "in Singapore. You DO NOT guess exact numeric resale prices; the "
                    "numeric prediction is done by a separate ML model. Focus on "
                    "concepts like factors affecting price, lease decay, town trends, etc."
                ),
            }
        ]

    # Display previous messages (skip the system one)
    for msg in st.session_state.chat_history[1:]:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

    # Input box at the bottom
    user_prompt = st.chat_input("Ask a question about HDB resale...")

    if user_prompt:
        # Show user message
        st.session_state.chat_history.append(
            {"role": "user", "content": user_prompt}
        )
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Ask OpenAI
        reply = ask_openai(st.session_state.chat_history)

        # Show assistant reply
        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply}
        )
        with st.chat_message("assistant"):
            st.markdown(reply)


# =====================================================================
#  SECTION 2: SINGLE-FLAT VALUATION FORM
# =====================================================================

with col_val:
    st.subheader("📈 Single HDB Price Estimate")

    st.write(
        "This section calls your ML pipeline (currently a stub returning a fixed "
        "value). Later, your data science team will plug in the real model."
    )

    with st.form("valuation_form"):
        town = st.text_input("Town", value="ANG MO KIO")
        flat_type = st.selectbox(
            "Flat type",
            options=["3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "OTHER"],
            index=2,
        )
        floor_area = st.number_input(
            "Floor area (sqm)",
            min_value=10.0,
            max_value=300.0,
            value=100.0,
            step=1.0,
        )
        storey_range = st.text_input("Storey range", value="10 TO 12")
        remaining_lease_years = st.number_input(
            "Remaining lease (years, approx)",
            min_value=0.0,
            max_value=99.0,
            value=60.0,
            step=1.0,
        )

        submitted = st.form_submit_button("Estimate Price")

    if submitted:
        features = {
            "town": town.strip(),
            "flat_type": flat_type.strip(),
            "floor_area_sqm": float(floor_area),
            "storey_range": storey_range.strip(),
            "remaining_lease_years": float(remaining_lease_years),
        }

        st.write("#### Features sent to ML model (debug)")
        st.json(features)

        estimated_price = predict_resale_price(features)

        st.success(
            f"**Estimated resale price:** ${estimated_price:,.0f}\n\n"
            "_Note: This is a model estimate, not an official valuation. "
            "Actual prices depend on flat condition, facing, proximity to amenities, "
            "and market sentiment._"
        )


st.markdown("---")


# =====================================================================
#  SECTION 3: BATCH VALUATION VIA CSV UPLOAD
# =====================================================================

st.header("📂 Batch HDB Valuation from CSV")

st.write(
    "Upload a CSV file containing multiple flats to get batch price estimates. "
    "Your CSV should have at least these columns: "
    "`town`, `flat_type`, `floor_area_sqm`, `storey_range`, `remaining_lease_years`."
)

uploaded_file = st.file_uploader(
    "Upload CSV file", type=["csv"], accept_multiple_files=False
)

required_cols = [
    "town",
    "flat_type",
    "floor_area_sqm",
    "storey_range",
    "remaining_lease_years",
]

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        df = None

    if df is not None:
        st.write("### Preview of uploaded data")
        st.dataframe(df.head())

        # Check required columns
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(
                "The following required columns are missing from your CSV: "
                + ", ".join(missing)
            )
        else:
            if st.button("Run Batch Predictions"):
                # Apply model row by row
                def predict_row(row):
                    features = {
                        "town": row["town"],
                        "flat_type": row["flat_type"],
                        "floor_area_sqm": float(row["floor_area_sqm"]),
                        "storey_range": row["storey_range"],
                        "remaining_lease_years": float(
                            row["remaining_lease_years"]
                        ),
                    }
                    return predict_resale_price(features)

                with st.spinner("Running predictions..."):
                    df["predicted_resale_price"] = df.apply(predict_row, axis=1)

                st.success("Batch predictions completed.")
                st.write("### Results")
                st.dataframe(df)

                # Allow download
                csv_out = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download results as CSV",
                    data=csv_out,
                    file_name="hdb_batch_predictions.csv",
                    mime="text/csv",
                )
else:
    st.info("Upload a CSV file to run batch predictions.")
