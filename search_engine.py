#DuckDuckGo Search se aapko latest live data mil jata hai.
import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain.agents import initialize_agent,AgentType
#StreamlitCallbackHandler allow karta hai ye real-time front-end me display karna:
#User ko har step ka reasoning dikhana
#Tool outputs live dikhana
#Final answer ke saath intermediate reasoning show karna
####
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

##Arix and Wikipedia tools
wiki_api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv_api_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki=WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
arxiv=ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

search=DuckDuckGoSearchRun(name="search")

st.title("Langchain eith chat with search")
#sidebar dor setting
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Grop Api key :",type="password")
if api_key:
     llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)

if "messages" not in st.session_state:
     st.session_state["messages"]=[
          {"role":"assistent","content":"Hi, how can i help you"}]
for msg in  st.session_state["messages"]:
     st.chat_message(msg["role"]).write(msg["content"])
if prompt:=st.chat_input(placeholder="what is machine learning"):
     st.session_state.messages.append({"role":"user","content":prompt})
     st.chat_message("user").write(prompt)
#Model partial tokens jaise jaise generate hote hain, aapko turant receive honge
     llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key,streaming=True)
     tools=[wiki,arxiv,search]
     
     search_agent=initialize_agent(tools,llm,AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)
     with st.chat_message("assistent"):
          st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
          response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
          st.session_state.messages.append({"role":"user","content":response})
          st.write(response)
