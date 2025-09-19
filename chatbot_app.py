import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
import json

# Model Initialization
device = torch.device("cpu")
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

st.set_page_config(page_title="Let's have a quick chat", layout="wide")

# Custom CSS styles
st.markdown("""
<style>
/* Page background */
[data-testid="stAppViewContainer"] { background-color: #000 !important; }
[data-testid="stMainContent"] { background-color: #000 !important; color: #fff; }
[data-testid="stSidebar"] { background-color: #111 !important; color: #fff; }

/* Chat logo & title */
.chat-logo { font-size: 2.5em; text-align: center; margin-bottom: 8px; }
.chat-title { color: #fff; font-size: 3em; font-weight: 800; text-align: center; margin-bottom: 6px; }

/* Chat container */
.chat-container { width: 100%; min-height: 450px; max-height: 70vh; overflow-y: auto; padding: 0 15px; border-radius: 0; }

/* Chat rows & bubbles */
.chat-row { display: flex; align-items: flex-end; margin-bottom: 28px; }
.bubble { padding: 18px 33px; border-radius: 28px; font-size: 1.5em; font-weight: 500; max-width: 60vw; word-break: break-word; }
.user-msg { background: #0d47a1; border-bottom-left-radius: 0; color: #fff; }
.bot-msg { background: #6a1b9a; border-bottom-right-radius: 0; color: #fff; }

/* Avatars */
.avatar { width: 62px; height: 62px; border-radius: 50%; font-size: 2.4em; background: #ffe066; display: flex; align-items: center; justify-content: center; margin: 0 19px; flex: none; }
.avatar-bot { background: #b2f2ff; margin-left: 19px; }

/* Timestamps */
.timestamp { font-size: 0.8em; color: #bbb; margin-top: 4px; }

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
    background-color: #1976f7 !important; color: white !important; font-weight: 600 !important;
    border-radius: 28px !important; padding: 10px 24px !important; width: 100%; margin-bottom: 12px !important;
    border: none !important; font-size: 1.1em !important; text-align: center;
}

/* Chat input */
[data-testid="stChat"] { background-color: #000 !important; }
.stChatInput { background-color: #000 !important; padding: 8px 0; }
.stChatInput textarea { background-color: #111 !important; color: #fff !important; border: 1px solid #444; border-radius: 12px !important; padding: 12px !important; }
.stChatInput textarea::placeholder { color: #bbb !important; opacity: 1; }
.stChatInput button { background-color: #1976f7 !important; color: #fff !important; font-weight: 600 !important; border-radius: 28px !important; padding: 8px 20px !important; border: none !important; }
</style>
""", unsafe_allow_html=True)

# Landing page logo and title
st.markdown('<div class="chat-logo">💬</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-title">Quick Chat!</div>', unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "reactions" not in st.session_state:
    st.session_state.reactions = [None] * len(st.session_state.messages)

# Capture reaction clicks from query parameters using new API
query_params = st.query_params
if "reaction" in query_params:
    value = query_params["reaction"][0]  # format: "index-emoji"
    idx, emoji = value.split("-")
    idx = int(idx)
    st.session_state.reactions[idx] = emoji
    st.set_query_params()  # clear query params

# Functions
def generate_response(user_input):
    chat_input = user_input + tokenizer.eos_token
    input_ids = tokenizer.encode(chat_input, return_tensors="pt").to(device)
    output_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    return tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

def clear_chat():
    st.session_state.messages = []
    st.session_state.reactions = []

def render_reactions(index):
    emojis = ["👍", "❤️", "😂"]
    reaction_html = ""
    for emoji in emojis:
        style = "font-weight:bold; font-size:1.2em;" if st.session_state.reactions[index] == emoji else "cursor:pointer; margin-right:6px;"
        reaction_html += f'<span style="{style}" onclick="window.location.href=\'?reaction={index}-{emoji}\'">{emoji}</span>'
    return reaction_html

# Chat input
if prompt := st.chat_input("Type your message..."):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
    st.session_state.reactions.append(None)
    with st.spinner("Thinking..."):
        response = generate_response(prompt)
        st.session_state.messages.append({"role": "bot", "content": response, "timestamp": timestamp})
        st.session_state.reactions.append(None)

# Chat container
st.markdown('<div class="chat-container" style="margin-top: -20px;">', unsafe_allow_html=True)

avatar_user = "🐶"
avatar_bot = "🐱"

# Display messages with timestamps and reactions
for i, msg in enumerate(st.session_state.messages):
    reaction_html = render_reactions(i)
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-row">
            <div class="avatar">{avatar_user}</div>
            <div>
                <div class="bubble user-msg">{msg['content']}</div>
                <div class="timestamp">{msg['timestamp']}</div>
                <div>{reaction_html}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-row" style="justify-content: flex-end;">
            <div>
                <div class="bubble bot-msg">{msg['content']}</div>
                <div class="timestamp">{msg['timestamp']}</div>
                <div>{reaction_html}</div>
            </div>
            <div class="avatar avatar-bot">{avatar_bot}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ✅ Conditional auto-scroll: only if there are messages
if st.session_state.messages:
    st.markdown("""
    <script>
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    </script>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Options")
st.sidebar.button("Clear Chat", on_click=clear_chat)

chat_json = json.dumps(st.session_state.messages, indent=2)
st.sidebar.download_button("Download Chat History", data=chat_json, file_name="chat_history.json")
