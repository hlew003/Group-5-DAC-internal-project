import os
import pandas as pd
import streamlit as st #streamlit for web app framework
from openai import OpenAI

# Load the ML model prediction function
import re
import joblib
from datetime import datetime
from pathlib import Path

# Resolve paths relative to this file so it works no matter where Streamlit is launched
BASE_DIR = Path(__file__).resolve().parent
DIST_LOOKUP_PATH = BASE_DIR / "geocode" / "distance_lookup.parquet"
MODEL_PATH = BASE_DIR / "geocode" / "rf_hdb_pipeline.pkl"

dist_lookup = pd.read_parquet(DIST_LOOKUP_PATH)
rf_pipeline = joblib.load(MODEL_PATH)

def parse_storey_range(storey_range_str):
    if pd.isna(storey_range_str):
        return np.nan
    s = str(storey_range_str).upper()
    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        return (int(nums[0]) + int(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    raise ValueError("storey_range must include at least one number, e.g. '01 TO 05'")

def build_feature_df(town, flat_type, storey_range, floor_area_sqm, remaining_lease_years, trans_year=None):
    base = pd.DataFrame([{
        "town": town,
        "flat_type": flat_type,
        "floor_area_sqm": floor_area_sqm,
        "remaining_lease_years": remaining_lease_years,
        "storey_avg": parse_storey_range(storey_range),
        "trans_year": trans_year or datetime.now().year,
    }])
    return base.merge(dist_lookup, on="town", how="left")

def predict_resale_price(inputs: dict) -> float:
    feat = build_feature_df(
        town=inputs["town"],
        flat_type=inputs["flat_type"],
        storey_range=inputs["storey_range"],
        floor_area_sqm=inputs["floor_area_sqm"],
        remaining_lease_years=inputs["remaining_lease_years"],
        trans_year=inputs.get("trans_year"),
    )
    return float(rf_pipeline.predict(feat)[0])


# ---------- LOAD API KEY  ----------

def load_api_key() -> str: #Function to load the OpenAI API Key, returns it as a string
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


api_key = load_api_key() #Load the OpenAI API key using the defined function
client = OpenAI(api_key=api_key) #Initialize the OpenAI client with the loaded API key (in string format)
OPENAI_MODEL = "gpt-4.1-mini"  # chosen model for chat completions, balancing performance and cost ('gpt-4.1-mini')


# ---------- OPENAI CHAT HELPER FUNCTION ----------

def ask_openai(messages): #Function to send a list of messages to OpenAI and receive a response
    """
    messages is a list of dicts: [{"role": "...", "content": "..."}]
    Returns assistant reply text.
    """
    response = client.chat.completions.create( 
        model=OPENAI_MODEL, #Model to be used for generating chat completions
        messages=messages, #List of message dicts representing the conversation history
        temperature=0.4, #Controls randomness in the output; lower values make output more focused and deterministic, higher values increase randomness
    )
    return response.choices[0].message.content #Return the content of the assistant's reply from the first choice


# ---------- STREAMLIT PAGE SETUP ----------

st.set_page_config( 
    page_title="HDB Resale Chatbot",
    page_icon="🏠", 
    layout="wide", #Use wide layout for better use of horizontal space
)

st.title("🏠 HDB Resale Chatbot & Valuation")
st.caption("Powered by OpenAI's GPT 4.1 Mini + DAA Group 5's Prediction Model")


# =====================================================================
#  SECTION 1: CHAT
# =====================================================================

st.header("💬 Chat about HDB Resale")

col_chat, col_val = st.columns([1, 1])  #Create two columns: chat area (1/3 width) and valuation form (2/3 width)

with col_chat: #Chat column 
    # Initialise chat history
    if "chat_history" not in st.session_state: #Check if chat history is already in session state
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
    for msg in st.session_state.chat_history[1:]: #Iterate through chat history, skipping the initial system message 
        with st.chat_message("user" if msg["role"] == "user" else "assistant"): #Display message as user or assistant based on role
            st.markdown(msg["content"]) #Render the message content in markdown format

    # Input box at the bottom
    user_prompt = st.chat_input("Ask a question about HDB resale...") #Create a chat input box for user questions

    if user_prompt: #When the user submits a prompt
        #Show user message
        st.session_state.chat_history.append( 
            {"role": "user", "content": user_prompt}
        ) #Append the user's message to the chat history
        with st.chat_message("user"):
            st.markdown(user_prompt) #Render the user's message in markdown format

        # Ask OpenAI
        reply = ask_openai(st.session_state.chat_history) #Call the ask_openai function with the current chat history to get the assistant's reply

        #Show assistant reply
        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply}
        ) #Append the assistant's reply to the chat history
        with st.chat_message("assistant"):
            st.markdown(reply) #Render the assistant's reply in markdown format


# =====================================================================
#  SECTION 2: SINGLE-FLAT VALUATION FORM
# =====================================================================

with col_val:
    st.subheader("📈 HDB Resale Price Estimate")

    st.write(
        "This section calls the ML pipeline (currently a stub returning a fixed "
        "value). To plug in ML Model later on."
    )

    with st.form("valuation_form"): #Create a form for HDB resale price estimation
        town = st.text_input("Town", value="ANG MO KIO") #Text input for town with default value
        flat_type = st.selectbox( #Dropdown select box for flat type
            "Flat Type",
            options=["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "OTHER"],
            index=3, #Default selection index 
        )
        floor_area = st.number_input( #Number input for floor area in square meters
            "Floor Area (sqm)",
            min_value=30.0,
            max_value=400.0,
            value=89.0,
            step=1.0,
        )
        storey_range = st.text_input("Storey Range (Format Example: 01 TO 03)", value="10 TO 12") #Text input for storey range with default value
        remaining_lease_years = st.number_input( #Number input for remaining lease years
            "Remaining Lease (approx. Years)",
            min_value=0.0,
            max_value=99.0,
            value=60.0,
            step=1.0,
        )

        submitted = st.form_submit_button("Estimate Price")

    if submitted: #When the form is submitted
        features = {
            "town": town.strip().upper(), 
            "flat_type": flat_type.strip(),
            "floor_area_sqm": float(floor_area),
            "storey_range": storey_range.strip().upper(),
            "remaining_lease_years": float(remaining_lease_years),
        }

        st.write("#### Features sent to ML model (debug)") #Debug section to display features sent to the ML model
        st.json(features) #Display the features in JSON format for debugging

        estimated_price = predict_resale_price(features) #Call the predict_resale_price function with the collected features to get the estimated price

        st.success(
            f"**Estimated resale price:** ${estimated_price:,.0f}\n\n"  
            "_Note: This is a model estimate, not an official valuation. "
            "Actual prices may vary, depending on flat condition, facing, proximity to amenities, "
            "and market sentiment._"
        )


st.markdown("---")  #Horizontal rule to separate sections


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

if uploaded_file is not None: #When a CSV file is uploaded
    try:
        df = pd.read_csv(uploaded_file) #Read the uploaded CSV file into a pandas DataFrame
    except Exception as e: #Handle exceptions that may occur while reading the CSV file
        st.error(f"Error reading CSV: {e}")
        df = None #Set df to None if there was an error

    if df is not None: #If the DataFrame was successfully created
        st.write("### Preview of uploaded data") #Section to preview the uploaded data
        st.dataframe(df.head()) #Display the first few rows of the DataFrame

        # Check required columns
        missing = [c for c in required_cols if c not in df.columns] #List comprehension to find any missing required columns in the DataFrame
        if missing:
            st.error(
                "The following required columns are missing from your CSV: "
                + ", ".join(missing) #Join the list of missing columns into a comma-separated string 
            )
        else:
            if st.button("Run Batch Predictions"): #Button to trigger batch predictions
                #Apply model row by row
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
                    return predict_resale_price(features) #Call the predict_resale_price function with the features extracted from the row

                with st.spinner("Running predictions..."): #Display a spinner while predictions are being run
                    df["predicted_resale_price"] = df.apply(predict_row, axis=1) #Apply the predict_row function to each row of the DataFrame and store the results in a new column

                st.success("Batch predictions completed.") #Display a success message when predictions are complete
                st.write("### Results") #Section to display the results
                st.dataframe(df) #Display the DataFrame with the predicted prices

                # Allow download
                csv_out = df.to_csv(index=False).encode("utf-8") #Convert the DataFrame to a CSV string without indexing, and encode it as UTF-8, standard [for CSV files]
                st.download_button(
                    label="Download results as CSV", #Button label for downloading the results as a CSV file
                    data=csv_out, #Data to be downloaded (the CSV string)
                    file_name="hdb_batch_predictions.csv", #Default filename for the downloaded CSV file
                    mime="text/csv", #selects the MIME type for the downloaded file
                )
else:
    st.info("Upload a CSV file to run batch predictions.") #Informational message prompting the user to upload a CSV file for batch predictions
