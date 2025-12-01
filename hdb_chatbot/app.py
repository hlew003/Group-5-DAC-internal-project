import os
from openai import OpenAI

from ml_stub import predict_resale_price  # later you'll swap this with the real ML file


# ---------- LOAD API KEY WITHOUT dotenv ----------

def load_api_key() -> str:
    """
    Load OPENAI_API_KEY from environment variable or from a .env file manually.
    This avoids using python-dotenv.
    """
    # 1) If it's already in environment variables, use that
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    # 2) Otherwise, try to read from .env in the current folder
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

    # 3) If still nothing, raise an error with a clear message
    raise RuntimeError(
        "OPENAI_API_KEY not found. "
        "Set it in your shell (export OPENAI_API_KEY=...) or in a .env file."
    )


api_key = load_api_key()
client = OpenAI(api_key=api_key)

# Choose a cheaper model to save money
OPENAI_MODEL = "gpt-4.1-mini"  # you can change this later if needed


# ---------- INTENT DETECTION ----------

def is_pricing_intent(text: str) -> bool:
    """
    Very simple way to guess if the user wants a PRICE / VALUATION.

    If the message contains any of the keywords below, we treat it
    as a pricing request and call the ML pipeline.
    """
    text = text.lower()
    keywords = [
        "predict price",
        "price prediction",
        "valuation",
        "estimate price",
        "resale price",
        "how much can i sell",
        "hdb valuation",
        "hdb price",
    ]
    return any(k in text for k in keywords)


# ---------- ASK USER FOR FLAT DETAILS ----------

def collect_flat_details() -> dict:
    """
    Ask the user questions in the terminal and build the 'features' dict
    that will be passed into predict_resale_price().

    IMPORTANT: The keys here must match what your ML team expects.
    """
    print("\n[HDB Valuation Mode] Please provide some details about the flat.")

    town = input("Town (e.g. 'ANG MO KIO'): ").strip()
    flat_type = input("Flat type (e.g. '3 ROOM', '4 ROOM', '5 ROOM'): ").strip()
    floor_area = float(input("Floor area (sqm): ").strip())
    storey_range = input("Storey range (e.g. '10 TO 12'): ").strip()
    remaining_lease_years = float(
        input("Remaining lease (years, approximate): ").strip()
    )

    features = {
        "town": town,
        "flat_type": flat_type,
        "floor_area_sqm": floor_area,
        "storey_range": storey_range,
        "remaining_lease_years": remaining_lease_years,
    }

    return features


# ---------- GENERAL CHAT WITH OPENAI ----------

def chat_with_openai(history, user_message: str) -> str:
    """
    Send normal (non-pricing) questions to OpenAI.
    'history' keeps previous messages so the bot remembers context.
    """
    messages = history + [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.4,
    )

    return response.choices[0].message.content


# ---------- MAIN LOOP ----------

def main():
    print("=== HDB ChatBot (CLI) ===")
    print("Type 'exit' to quit.")
    print("If you want a valuation, use words like 'valuation', 'resale price', etc.\n")

    # Initial system message tells GPT how to behave for general chat
    history = [
        {
            "role": "system",
            "content": (
                "You are an assistant that answers questions about HDB resale flats "
                "in Singapore. For price estimation, a separate ML model is used "
                "by the backend; you only handle general explanations and advice."
            ),
        }
    ]

    while True:
        user_msg = input("You: ").strip()
        if user_msg.lower() == "exit":
            print("Goodbye.")
            break

        # If we detect pricing intent, use ML stub / ML pipeline
        if is_pricing_intent(user_msg):
            features = collect_flat_details()
            estimated_price = predict_resale_price(features)

            bot_reply = (
                f"Based on your inputs, the estimated resale price is "
                f"around ${estimated_price:,.0f}.\n"
                "(Note: this is a model estimate, not an official valuation.)"
            )

            print("\nBot:", bot_reply, "\n")
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": bot_reply})

        else:
            # Normal chat goes to OpenAI
            bot_reply = chat_with_openai(history, user_msg)
            print("\nBot:", bot_reply, "\n")

            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": bot_reply})


if __name__ == "__main__":
    main()
