import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import pickle
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema import HumanMessage

# Load environment variables from .env
load_dotenv()

# Load the book dataset
df = pd.read_csv("processed_data.csv")

# Load the pre-trained SVD model for recommendations
with open("svd_model.pkl", "rb") as f:
    svd_model = pickle.load(f)

# Initialize Hugging Face Chatbot
llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")
chat_model = ChatHuggingFace(llm=llm)

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# App Layout
app.layout = dbc.Container([
    html.H1("📚 AI-Powered Library System", className="text-center text-primary mb-4"),

    # Book Recommendation Section
    html.H3("📖 Get Book Recommendations:", className="text-white"),
    dbc.Row([
        dbc.Col(dcc.Input(id="user_id", type="text", placeholder="Enter User ID", className="form-control"), width=6),
        dbc.Col(dbc.Button("Recommend", id="recommend_button", color="primary"), width=2),
    ]),
    html.Div(id="recommend_output", className="text-white mt-3"),

    html.Hr(),

    # AI Chatbot Section
    html.H3("🤖 Chat with AI Library Assistant", className="text-white"),
    dbc.Row([
        dbc.Col(dcc.Textarea(id="user_message", placeholder="Type your message...", className="form-control", rows=3), width=8),
        dbc.Col(dbc.Button("Send", id="chat_button", color="success", className="mt-2"), width=2),
    ]),
    html.Div(id="chat_output", className="text-white mt-3"),
], fluid=True)

# Book Recommendation Callback
@app.callback(
    Output("recommend_output", "children"),
    Input("recommend_button", "n_clicks"),
    State("user_id", "value")
)
def recommend_books(n_clicks, user_id):
    if not user_id:
        return "⚠️ Please enter a valid User ID."

    try:
        book_ids = df["Clean_Title"].unique()
        predictions = [(book, svd_model.predict(user_id, book).est) for book in book_ids]
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

        if recommendations:
            return f"📚 Recommended Books: {', '.join([book for book, _ in recommendations])}"
        else:
            return "❌ No recommendations found."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Chatbot Callback
@app.callback(
    Output("chat_output", "children"),
    Input("chat_button", "n_clicks"),
    State("user_message", "value")
)
def chatbot_response(n_clicks, message):
    if not message:
        return "❓ Please enter a message."
    
    try:
        response = chat_model.invoke([HumanMessage(content=message)])
        chatbot_reply = response.content
    except Exception as e:
        chatbot_reply = f"⚠️ Error: {str(e)}"

    return f"🤖 **AI:** {chatbot_reply}"

if __name__ == "__main__":
    app.run(debug=True)
