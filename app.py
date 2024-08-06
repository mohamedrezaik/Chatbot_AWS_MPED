import requests
import streamlit as st
from PIL import Image
import json
import uuid
from langchain_core.messages import AIMessage, HumanMessage
from invoke_agent import SqlAgent

st.set_page_config(
    page_title="Chat With MPED",
    page_icon=":speech_balloon:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Chat With MPED")

def generate_new_session_id():
    return str(uuid.uuid4())

if 'sessionId' not in st.session_state:
    st.session_state['sessionId'] = generate_new_session_id()
    st.session_state['sql_agent'] = SqlAgent()

def fetch_data(question):

    try: 
        # Subracting 0 index represnts the response of the agent
        response = st.session_state['sql_agent'].invoke_agent(question)[0].get("output")
        print(response)
        return response
    except Exception as e:
        print(f"Failed to invoke agent: {e}")
        return "Hi! Could you please repeat your question or provide more details? Thanks!"

# st.set_page_config(page_title="ChatBot with DATABASE", page_icon=":speech_balloon:")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
local_css("style.css")


# Custom CSS for sidebar background color
custom_sidebar_css = """
<style>
    [data-testid="stSidebar"] {
        background-color: #0275d8;
    }
</style>
"""
st.markdown(custom_sidebar_css, unsafe_allow_html=True)
with st.sidebar:
    st.image("logo.png", use_column_width=True)
    st.write("")
    st.write("") 
    st.write("") 
    st.write("") 
    st.write("") 

    
    custom_subheader_css = """
    <style>
        .custom-subheader { color: #ffffff; }
    </style>
    """
    st.markdown(custom_subheader_css, unsafe_allow_html=True)
    st.markdown('<h3 class="custom-subheader">Overview</h3>', unsafe_allow_html=True)
    st.write("""<p style="text-align: left;">This is an AI ChatBot to ask it about National Accounts Data of Egypt from The Ministry of Planning and Economic Development.</p>""", unsafe_allow_html=True)    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    custom_button_css = """
    <style>
        .stButton>button {
            background-color: #ffffff;
            font-weight: bold;
            color: black;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-color= #CCBEBE;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #CCBEBE;
        }
    </style>
    """
    custom_css = """
        <style>
            .stAlert {
                background-color: transparent !important;
                color: white !important;
                border: none !important;
            }
        </style>
        """

    # Apply the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown(custom_button_css, unsafe_allow_html=True)
    if st.button("Clear"):
        with st.spinner("Clearing Session..."):
            st.session_state['sessionId'] = generate_new_session_id()
            st.session_state['sql_agent'].clear_chat_history()
            st.session_state["chat_history"] = [
                AIMessage(content="Hello! I am a AI assistant. Ask me anything about the National Accounts Data of Egypt.")
            ]
            st.success("Session Cleared Successfully!")
            # st.markdown(f'<p class="custom-success">{data}</p>', unsafe_allow_html=True)

# Additional custom CSS for chat bubble
custom_chat_css = """
<style>
    .chat-bubble {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-bubble {
        background-color: #f1f0f0;
        color: black;
    }
</style>
"""
# st.markdown(custom_chat_css, unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        AIMessage(content="Hello! I am a AI assistant. Ask me anything about the National Accounts Data of Egypt.")
    ]

for msg in st.session_state["chat_history"]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("Human"):
                # st.write(f'<div class="chat-bubble user-bubble">{msg.content}</div>', unsafe_allow_html=True)
                # st.markdown(f"""<div style="text-align: left; width: 100%;">{msg.content}</div>""", unsafe_allow_html=True)
                # st.markdown(msg.content, unsafe_allow_html=True)
                st.markdown(f'<div style="text-align": left; width: 120%; color:black !important>{msg.content}</div>', unsafe_allow_html=True) 
        elif isinstance(msg, AIMessage):
            with st.chat_message("AI"):
                # st.write(f'<div class="chat-bubble">{msg.content}</div>', unsafe_allow_html=True)
                # st.markdown(f"""<div style="text-align: left; width: 100%;">{msg.content}</div>""", unsafe_allow_html=True)
                with st.expander("AI Output", expanded=True):
                    # st.markdown(msg.content, unsafe_allow_html=True)
                    st.markdown(f'<div style="text-align": left; width: 120%; color:black !important>{msg.content}</div>', unsafe_allow_html=True)    

user_query = st.chat_input("Type a message...")
if user_query:
    st.session_state["chat_history"].append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        # st.write(f'<div class="chat-bubble user-bubble">{user_query}</div>', unsafe_allow_html=True)
        # st.markdown(f"""<div style="text-align: left; width: 100%;">{user_query}</div>""", unsafe_allow_html=True)    
        # st.markdown(user_query, unsafe_allow_html=True)
        st.markdown(f'<div style="text-align": left; width: 120%; color:black !important>{user_query}</div>', unsafe_allow_html=True) 
    
    with st.chat_message("AI"):
        with st.spinner("Fetching response..."):
            response_content = fetch_data(user_query)
            if response_content and response_content.strip() != "":
                st.session_state["chat_history"].append(AIMessage(content=response_content))
                # st.write(f'<div class="chat-bubble">{response_content}</div>', unsafe_allow_html=True)
                # st.markdown(f"""<div style="text-align: left; width: 100%;">{response_content}</div>""", unsafe_allow_html=True)
                with st.expander("AI Output", expanded=True):
                    # st.markdown(response_content, unsafe_allow_html=True)
                    st.write(f'<div style="text-align": left; width: 120%; color:black !important>{response_content}</div>', unsafe_allow_html=True)
    # Rerun the script to update the chat display
    st.rerun()