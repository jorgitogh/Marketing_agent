# app.py
# 1-click Marketing Creative: PRO prompt + COPY (Groq) + IMAGE (SiliconFlow FLUX)
#
# pip install -U streamlit requests pillow python-dotenv langchain-groq langchain-core
# streamlit run app.py

from __future__ import annotations

import os
import base64
from io import BytesIO
from typing import Optional, Dict, Any, Tuple

import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

load_dotenv()

# ---------------- UI ----------------
st.set_page_config(page_title="Marketing Creative Generator", page_icon="🧠", layout="wide")
st.title("🧠 Marketing Creative Generator (1-click)")
st.caption("Un botón: genera **copy + prompt pro + imagen**. Texto con Groq, imagen con SiliconFlow (FLUX).")

# ---------------- Defaults ----------------
PLATFORM_DEFAULT_AR = {
    "Instagram": "4:5",
    "Facebook": "4:5",
    "LinkedIn": "1.91:1",
    "Blog": "16:9",
    "E-mail": "16:9",
}

ASPECT_OPTIONS = ["1:1", "4:5", "3:4", "4:3", "9:16", "16:9", "1.91:1"]

# SiliconFlow supports sizes like 512x512 etc. Map aspect ratio -> size.
AR_TO_SIZE = {
    "1:1": "1024x1024",
    "4:5": "1024x1280",
    "3:4": "1024x1365",
    "4:3": "1280x960",
    "9:16": "1080x1920",
    "16:9": "1920x1080",
    "1.91:1": "1910x1000",
}

SILICONFLOW_URL = "https://api.siliconflow.com/v1/images/generations"

# ---------------- LangChain: PRO prompt + COPY prompt ----------------
TEXT_CHAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior marketing creative director. "
     "You write persuasive Spanish marketing copy and craft excellent image prompts."),
    ("human",
     """
Create:
A) COPY in Spanish for {platform} (tone: {tone}, audience: {audience}).
B) FINAL_IMAGE_PROMPT (English): ONE detailed image-generation prompt for a premium marketing visual.

USER INTENT (simple):
{simple_prompt}

CONTEXT
- Platform: {platform}
- Target audience: {audience}
- Tone: {tone}
- Visual style: {style}
- Aspect ratio: {aspect_ratio}
- Brand name: {brand_name}
- Brand colors: {brand_colors}
- CTA: {cta}
- Hashtags: {hashtags}

RULES
- COPY: Spanish, platform-appropriate, no quotes. If CTA enabled, include CTA at end.
  If hashtags enabled and platform is Instagram/Facebook, add 5–10 hashtags at end.
- FINAL_IMAGE_PROMPT: Must specify composition, lighting, background, focal subject,
  brand colors, style cues. Add a final line starting with "NEGATIVE:" listing what to avoid:
  clutter, watermarks, distorted hands, low-res artifacts, gibberish text.

Return EXACTLY:
A) COPY:
...

B) FINAL_IMAGE_PROMPT:
...
Extra notes: {extra}
     """.strip())
])

def parse_copy_and_imageprompt(text: str) -> Tuple[str, str]:
    copy = text.strip()
    img_prompt = ""
    if "A) COPY:" in text and "B) FINAL_IMAGE_PROMPT:" in text:
        part = text.split("A) COPY:", 1)[1]
        a, b = part.split("B) FINAL_IMAGE_PROMPT:", 1)
        copy = a.strip()
        img_prompt = b.strip()
    return copy, img_prompt

def build_text_chain(groq_api_key: str, model: str = "llama-3.3-70b-versatile"):
    llm = ChatGroq(
        model=model,
        api_key=groq_api_key,
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    chain = TEXT_CHAIN_PROMPT | llm | StrOutputParser()
    return chain

# ---------------- SiliconFlow image generation ----------------
def siliconflow_generate_image(
    token: str,
    model: str,
    prompt: str,
    image_size: str,
) -> bytes:
    """
    Calls SiliconFlow images generation endpoint.
    Returns raw image bytes (png/jpg) by decoding base64 if needed.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "image_size": image_size,
    }

    r = requests.post(SILICONFLOW_URL, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    # SiliconFlow responses may vary. Common patterns:
    # - data["data"][0]["b64_json"]
    # - data["images"][0]["base64"]
    # - data["data"][0]["url"] (then download)
    #
    # We'll handle the typical b64 case and fallback to URL if present.
    b64 = None
    url = None

    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list) and data["data"]:
            item = data["data"][0]
            if isinstance(item, dict):
                b64 = item.get("b64_json") or item.get("base64") or item.get("image_base64")
                url = item.get("url")
        if not b64 and "images" in data and isinstance(data["images"], list) and data["images"]:
            item = data["images"][0]
            if isinstance(item, dict):
                b64 = item.get("b64_json") or item.get("base64") or item.get("image_base64")
                url = item.get("url")

    if b64:
        return base64.b64decode(b64)

    if url:
        img_r = requests.get(url, timeout=120)
        img_r.raise_for_status()
        return img_r.content

    raise ValueError(f"Could not find image bytes in response: keys={list(data.keys())}")

# ---------------- LangChain pipeline (1 click) ----------------
def full_pipeline(inputs: Dict[str, Any]) -> Dict[str, Any]:
    # 1) Generate COPY + FINAL_IMAGE_PROMPT with Groq
    text_chain = build_text_chain(inputs["groq_api_key"], model=inputs["groq_model"])
    text_out = text_chain.invoke({
        "platform": inputs["platform"],
        "tone": inputs["tone"],
        "audience": inputs["audience"],
        "style": inputs["style"],
        "aspect_ratio": inputs["aspect_ratio"],
        "brand_name": inputs["brand_name"] or "None",
        "brand_colors": inputs["brand_colors"] or "None",
        "cta": "Yes" if inputs["cta"] else "No",
        "hashtags": "Yes" if inputs["hashtags"] else "No",
        "simple_prompt": inputs["simple_prompt"],
        "extra": inputs["extra"] or "(none)",
    })

    copy, img_prompt = parse_copy_and_imageprompt(text_out)

    # 2) Generate IMAGE with SiliconFlow
    image_size = inputs["image_size"]
    if image_size == "auto":
        image_size = AR_TO_SIZE.get(inputs["aspect_ratio"], "1024x1024")

    img_bytes = siliconflow_generate_image(
        token=inputs["siliconflow_token"],
        model=inputs["siliconflow_model"],
        prompt=img_prompt if img_prompt else inputs["simple_prompt"],
        image_size=image_size,
    )

    return {
        "copy": copy,
        "final_image_prompt": img_prompt,
        "image_bytes": img_bytes,
        "raw_text": text_out,
        "image_size": image_size,
    }

chain = (
    RunnablePassthrough()
    | RunnableLambda(full_pipeline)
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("🔑 Keys")
    groq_key = st.text_input("Groq API Key (texto)", type="password", placeholder="gsk_...")
    siliconflow_token = st.text_input("SiliconFlow Token (imagen)", type="password", placeholder="sf_...")

    st.divider()
    st.subheader("⚙️ Models")
    groq_model = st.selectbox("Groq model", ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile"], index=0)
    siliconflow_model = st.text_input("SiliconFlow image model", value="black-forest-labs/FLUX.2-pro")

    st.divider()
    st.subheader("🖼️ Image size")
    image_size = st.selectbox("Size", ["auto", "512x512", "768x768", "1024x1024", "1024x1280", "1080x1920", "1920x1080"], index=0)
    st.caption("En auto, se elige según aspect ratio.")

# ---------------- Front ----------------
left, right = st.columns([1.05, 1])

with left:
    st.subheader("🧩 Inputs")

    platform = st.selectbox("Platform", ["Instagram", "Facebook", "LinkedIn", "Blog", "E-mail"])
    default_ar = PLATFORM_DEFAULT_AR.get(platform, "1:1")
    default_idx = ASPECT_OPTIONS.index(default_ar) if default_ar in ASPECT_OPTIONS else 0
    aspect_ratio = st.selectbox("Aspect ratio", ASPECT_OPTIONS, index=default_idx)

    c1, c2 = st.columns(2)
    with c1:
        audience = st.selectbox("Audience", ["All", "Young adults", "Families", "Seniors", "Teenagers"])
        tone = st.selectbox("Tone", ["Normal", "Informative", "Inspiring", "Urgent", "Informal"])
    with c2:
        style = st.selectbox(
            "Visual style",
            ["Photorealistic", "Minimalist", "3D clean render", "Flat illustration", "Modern collage", "Product mockup"],
        )

    brand_name = st.text_input("Brand name (optional)", placeholder="e.g., Outdoorclothes")
    brand_colors = st.text_input("Brand colors (optional)", placeholder="e.g., #0B1F3B, white")

    cta = st.checkbox("Include CTA", value=True)
    hashtags = st.checkbox("Add hashtags (IG/FB)", value=True)

    simple_prompt = st.text_area(
        "Prompt simple (qué quieres promocionar)",
        placeholder="Ej: Promocionar una chaqueta impermeable premium para navegar. Estética elegante y moderna.",
        height=110
    )
    extra = st.text_area(
        "Notas extra (opcional)",
        placeholder="Ej: Luz cálida de atardecer en barco, fondo marino, look premium, espacio para texto arriba.",
        height=90
    )

    generate = st.button("🚀 Generar (copy + imagen)", use_container_width=True)

with right:
    st.subheader("📤 Output")
    out = st.empty()

# ---------------- Action ----------------
if generate:
    if not groq_key:
        st.error("Introduce tu Groq API Key (texto).")
        st.stop()
    if not siliconflow_token:
        st.error("Introduce tu SiliconFlow token (imagen).")
        st.stop()
    if not simple_prompt.strip():
        st.error("Escribe un prompt simple.")
        st.stop()

    with st.spinner("Generando copy + prompt pro + imagen..."):
        try:
            result = chain.invoke({
                "groq_api_key": groq_key,
                "groq_model": groq_model,
                "siliconflow_token": siliconflow_token,
                "siliconflow_model": siliconflow_model,
                "image_size": image_size,
                "platform": platform,
                "aspect_ratio": aspect_ratio,
                "audience": audience,
                "tone": tone,
                "style": style,
                "brand_name": brand_name,
                "brand_colors": brand_colors,
                "cta": cta,
                "hashtags": hashtags,
                "simple_prompt": simple_prompt,
                "extra": extra,
            })
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    with out.container():
        # Image
        img_bytes = result.get("image_bytes")
        if img_bytes:
            st.subheader("🖼️ Imagen")
            img = Image.open(BytesIO(img_bytes))
            st.image(img, use_container_width=True)
            st.download_button("Download image", img_bytes, "creative.png", "image/png", use_container_width=True)
            st.caption(f"Image size used: {result.get('image_size')}")
        else:
            st.warning("No he recibido imagen. Mira el debug.")
            with st.expander("Debug"):
                st.write(result)

        # Copy
        st.subheader("✍️ Copy")
        st.markdown(result.get("copy", ""))

        with st.expander("🧾 Prompt pro de imagen usado"):
            st.code(result.get("final_image_prompt", ""), language="text")

        with st.expander("🧾 Raw (Groq output)"):
            st.code(result.get("raw_text", ""), language="text")
