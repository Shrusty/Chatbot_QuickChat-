A simple chatbot web app built using **Streamlit** and **Microsoft DialoGPT-small**.  
It supports styled chat bubbles, dark mode UI, emojis for reactions, and chat history clear and download.


---
## Features
-  Dark themed chat window with styled bubbles  
- 🐶 User and 🐱 Bot avatars  
- Emoji reactions for each message  
- Download chat history as JSON  
- Clear chat option  
- Powered by HuggingFace `microsoft/DialoGPT-small`

https://snvfvscrbhn468vjt8pwex.streamlit.app/


  ## Installation

1. Clone the repository:
   git clone https://github.com/your-username/my_chatbot.git
   cd my_chatbot
   
2. Create and activate a virtual environment:
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows

3. Install dependencies:
pip install -r requirements.txt

4. Run the Streamlit app:
 streamlit run chatbot_app.py

## Tech Stack
Streamlit – UI framework

HuggingFace Transformers – DialoGPT model

PyTorch – Model backend


## License
This project is open-source and available under the MIT License.
