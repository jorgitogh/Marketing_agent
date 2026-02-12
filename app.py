import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ---------------- UI ----------------
st.set_page_config(page_title="Content Generator 🤖", page_icon="🤖")
st.title("Content generator")

with st.sidebar:
    st.subheader("🔑 Groq API Key")
    api_key = st.text_input("API Key", type="password", placeholder="gsk_...")
    st.caption("Consíguela en la consola de Groq.")
    st.divider()

# ---------------- Form ----------------
topic = st.text_input("Topic:", placeholder="e.g., nutrition, mental health, routine check-ups...")
platform = st.selectbox("Platform:", ['Instagram', 'Facebook', 'LinkedIn', 'Blog', 'E-mail'])
tone = st.selectbox("Message tone:", ['Normal', 'Informative', 'Inspiring', 'Urgent', 'Informal'])
length = st.selectbox("Text length:", ['Short', 'Medium', 'Long'])
audience = st.selectbox("Target audience:", ['All', 'Young adults', 'Families', 'Seniors', 'Teenagers'])
cta = st.checkbox("Include CTA")
hashtags = st.checkbox("Return Hashtags")
keywords = st.text_area("Keywords (SEO):", placeholder="Example: wellness, preventive healthcare...")

# ---------------- LLM chain ----------------
def llm_generate(llm, prompt: str) -> str:
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a digital marketing expert specialized in SEO and persuasive copywriting."),
        ("human", "{prompt}"),
    ])
    chain = template | llm | StrOutputParser()
    return chain.invoke({"prompt": prompt})


# ---------------- Action ----------------
if st.button("Content generator"):
    if not api_key:
        st.error("Introduce tu Groq API Key en la barra lateral para continuar.")
        st.stop()

    # Instancia el LLM SOLO cuando ya tienes la key
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    prompt = f"""
Write an SEO-optimized text on the topic '{topic}'.
Return only the final text in your response and don't put it inside quotes.
- Platform where it will be published: {platform}.
- Tone: {tone}.
- Target audience: {audience}.
- Length: {length}.
- {"Include a clear Call to Action." if cta else "Do not include a Call to Action."}
- {"Include relevant hashtags at the end of the text." if hashtags else "Do not include hashtags."}
{"- Keywords to include (for SEO): " + keywords if keywords else ""}
""".strip()

    try:
        res = llm_generate(llm, prompt)
        st.markdown(res)
    except Exception as e:
        st.error(f"Error: {e}")
