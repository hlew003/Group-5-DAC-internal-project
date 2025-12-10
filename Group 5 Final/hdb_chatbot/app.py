import os #Allows interaction with the operating system for environment variables (such as API keys which shouldbe stored outside the code))
from openai import OpenAI #import the OpenAI client library to interact with OpenAI's API

from ml_stub import predict_resale_price  # local module simulating ML model for HDB price prediction

# ---------- OPENAI API SETUP ----------
def load_api_key() -> str: #Function to load the OpenAI API key from environment variables or a .env file
    """
    Load OPENAI_API_KEY from environment variable or from a .env file manually.
    This avoids using python-dotenv. 
    """
    # 1) If it's already in environment variables, use that 
    api_key = os.environ.get("OPENAI_API_KEY") #get function on 'os.environ'
    if api_key: #If the api_key variable is not None or Empty
        return api_key 

    # 2) Otherwise, try to read from .env in the current folder
    try: #Try block to handle potential file not found error 
        with open(".env", "r") as f: #Open the .env file for reading ('r') 
            for line in f: #Iterate through each line in the file
                line = line.strip() #Remove leading/trailing whitespace
                if not line or line.startswith("#"): #Skip empty lines or comments
                    continue #Continue to the next line
                if line.startswith("OPENAI_API_KEY="): #Check if the line contains the API key
                    return line.split("=", 1)[1].strip() #Extract and return the API key value
    except FileNotFoundError: #If the .env file is not found, pass and move on
        pass 

    # 3) If still nothing, raise an error with a clear message
    raise RuntimeError(
        "OPENAI_API_KEY not found. "
        "Set it in your shell (export OPENAI_API_KEY=...) or in a .env file." #Error message guiding the user to set the API key
    )


api_key = load_api_key() #Load the OpenAI API key using the defined function
client = OpenAI(api_key=api_key) #Initialize the OpenAI client with the loaded API key

# Set the OpenAI model to be used for chat completions
OPENAI_MODEL = "gpt-4.1-mini"  # Selected "gpt-4.1-mini" for a balance of performance and cost


# ---------- CHAT INTENT DETECTION FUNCTION ----------

def is_pricing_intent(text: str) -> bool: #Function to determine if the user's message indicates a request for pricing or valuation
    """
    Very simple way to guess if the user wants a PRICE / VALUATION.

    If the message contains any of the keywords below, we treat it
    as a pricing request and call the ML pipeline.
    """
    text = text.lower() #Convert the input text to lowercase for case-insensitive matching
    keywords = [
        "predict price",
        "price prediction",
        "valuation",
        "estimate price",
        "resale price",
        "how much can i sell",
        "hdb valuation",
        "hdb price",
    ] #List of keywords indicating pricing intent
    return any(k in text for k in keywords) #Return True if any keyword is found in the text, otherwise False


# ---------- ASK USER FOR FLAT DETAILS ----------

def collect_flat_details() -> dict: #Function to interactively collect flat details from the user for price prediction
    """
    Ask the user questions in the terminal and build the 'features' dict
    that will be passed into predict_resale_price().

    IMPORTANT: The keys here must match what your ML team expects.
    """
    print("\n[HDB Valuation Mode] Please provide some details about the flat.")

    town = input("Town (e.g. 'ANG MO KIO'): ").strip()  #Get the town input from the user and remove leading/trailing whitespace
    flat_type = input("Flat type (e.g. '3 ROOM', '4 ROOM', '5 ROOM'): ").strip() #Get the flat type input from the user and remove leading/trailing whitespace
    floor_area = float(input("Floor area (sqm): ").strip()) #Get the floor area input from the user, remove whitespace, and convert to float
    storey_range = input("Storey range (e.g. '10 TO 12'): ").strip() #Get the storey range input from the user and remove leading/trailing whitespace
    remaining_lease_years = float( #Get the remaining lease years input from the user, remove whitespace, and convert to float
        input("Remaining lease (years, approximate): ").strip()
    )

    features = { #Build a dictionary of features to be used for price prediction
        "town": town, #Store the town in the features dictionary
        "flat_type": flat_type, #Store the flat type in the features dictionary
        "floor_area_sqm": floor_area, #Store the floor area in square meters in the features dictionary
        "storey_range": storey_range, #Store the storey range in the features dictionary
        "remaining_lease_years": remaining_lease_years, #Store the remaining lease years in the features dictionary
    }

    return features #Return the features dictionary


# ---------- GENERAL CHAT FUNCTION WITH OPENAI ----------

def chat_with_openai(history, user_message: str) -> str: #Function to handle general chat interactions with OpenAI, maintaining conversation history
    """
    Send normal (non-pricing) questions to OpenAI.
    'history' keeps previous messages so the bot remembers context.
    """
    messages = history + [{"role": "user", "content": user_message}] #Combine conversation history with the new user message

    response = client.chat.completions.create( #Call the OpenAI API to create a chat completion
        model=OPENAI_MODEL, #Specify the model to be used for the chat completion
        messages=messages, #Pass the combined messages to the API
        temperature=0.4, #Set the temperature for response variability - at 0.4, responses are balanced between creativity and determinism.
    )

    return response.choices[0].message.content #Return the content of the first message choice from the API response


# ---------- MAIN LOOP ----------

def main():
    print("=== HDB ChatBot (CLI) ===") #Print the title of the chatbot application
    print("Type 'exit' to quit.") #Print instructions for exiting the application
    print("If you want a valuation, use words like 'valuation', 'resale price', etc.\n") #Print instructions for requesting a valuation

    # Initial system message tells GPT how to behave for general chat
    history = [
        {
            "role": "system", #System message defining the assistant's role and limitations
            "content": (
                "You are an assistant that answers questions about HDB resale flats " #Define the assistant's role and limitations
                "in Singapore. For price estimation, a separate ML model is used " #Indicate that a separate ML model is used for price estimation
                "by the backend; you only handle general explanations and advice." #Clarify that the assistant handles general explanations and advice
            ),
        }
    ]

    while True: #Start an infinite loop for continuous user interaction - will break on 'exit' command 
        user_msg = input("You: ").strip() #Prompt the user for input and remove leading/trailing whitespace
        if user_msg.lower() == "exit": #Check if the user wants to exit the application
            print("Goodbye.") #Print a goodbye message
            break #Exit the loop and end the application

        # If we detect pricing intent, use ML stub / ML pipeline
        if is_pricing_intent(user_msg): #Check if the user's message indicates a pricing intent
            features = collect_flat_details() #Collect flat details from the user for price prediction
            estimated_price = predict_resale_price(features) #Call the ML model to predict the resale price based on the collected features

            bot_reply = ( #Construct the bot's reply with the estimated price
                f"Based on your inputs, the estimated resale price is " #Format the estimated price with commas and two decimal places
                f"around ${estimated_price:,.2f}.\n" 
                "(Note: this is a model estimate, not an official valuation.)" #Add a disclaimer about the estimate
            )

            print("\nBot:", bot_reply, "\n") #Print the bot's reply
            history.append({"role": "user", "content": user_msg}) #Update the conversation history with the user's message
            history.append({"role": "assistant", "content": bot_reply}) #Update the conversation history with the bot's reply

        else:
            # Normal chat goes to OpenAI 
            bot_reply = chat_with_openai(history, user_msg) #Handle general chat interactions with OpenAI
            print("\nBot:", bot_reply, "\n") #Print the bot's reply

            history.append({"role": "user", "content": user_msg}) #Update the conversation history with the user's message
            history.append({"role": "assistant", "content": bot_reply}) #Update the conversation history with the bot's reply


if __name__ == "__main__": #Run the main function if this script is executed directly
    main() #Call the main function to start the chatbot application
