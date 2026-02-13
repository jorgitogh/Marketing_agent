# app.py
# 1-click: Inputs -> Pro prompt -> Gemini TEXT (copy + final_image_prompt) -> Gemini IMAGE -> show image + copy
#
# pip install -U streamlit google-genai pillow langchain-core
# streamlit run app.py

from __future__ import annotations

from io import BytesIO
from typing import Optional, Tuple, Dict, Any

import streamlit as st
from PIL import Image

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from google import genai
from google.genai import types


# ---------------- UI ----------------
st.set_page_config(page_title="Marketing Creative Generator", page_icon="🧠", layout="wide")
st.title("🧠 Marketing Creative Generator (1-click)")
st.caption("Un botón: genera **copy + prompt pro + imagen** con Gemini.")


# ---------------- Defaults ----------------
PLATFORM_DEFAULT_AR = {
    "Instagram": "4:5",
    "Facebook": "4:5",
    "LinkedIn": "1.91:1",
    "Blog": "16:9",
    "E-mail": "16:9",
}
ASPECT_OPTIONS = ["1:1", "4:5", "3:4", "4:3", "9:16", "16:9", "1.91:1"]


# ---------------- Extractors ----------------
def extract_text(resp) -> str:
    t = getattr(resp, "text", None)
    if t:
        return t
    if not resp or not getattr(resp, "candidates", None):
        return ""
    for cand in resp.candidates:
        content = getattr(cand, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for p in content.parts:
            pt = getattr(p, "text", None)
            if pt:
                return pt
    return ""


def extract_first_image_bytes(resp) -> Optional[bytes]:
    if not resp or not getattr(resp, "candidates", None):
        return None
    for cand in resp.candidates:
        content = getattr(cand, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for p in content.parts:
            inline = getattr(p, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return inline.data
    return None


def parse_copy_and_imageprompt(text: str) -> Tuple[str, str]:
    """
    Expects:
      A) COPY:
      ...
      B) FINAL_IMAGE_PROMPT:
      ...
    """
    copy = text.strip()
    img_prompt = ""
    if "A) COPY:" in text and "B) FINAL_IMAGE_PROMPT:" in text:
        part = text.split("A) COPY:", 1)[1]
        a, b = part.split("B) FINAL_IMAGE_PROMPT:", 1)
        copy = a.strip()
        img_prompt = b.strip()
    return copy, img_prompt


# ---------------- LangChain prompts ----------------
TEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior marketing creative director. "
     "You write persuasive Spanish marketing copy and craft excellent image-generation prompts."),
    ("human",
     """
Create:
A) COPY in Spanish for {platform} (tone: {tone}, audience: {audience}).
B) FINAL_IMAGE_PROMPT (English): a single, detailed image-generation prompt for a premium marketing visual.

USER INTENT (simple):
{simple_prompt}

CONTEXT
- Platform: {platform}
- Aspect ratio: {aspect_ratio}
- Target audience: {audience}
- Tone: {tone}
- Visual style: {style}
- Brand name: {brand_name}
- Brand colors: {brand_colors}
- CTA: {cta}
- Hashtags: {hashtags}

RULES
- COPY: Spanish, platform-appropriate, no quotes. If CTA enabled, include CTA at end.
  If hashtags enabled and platform is Instagram/Facebook, add 5–10 hashtags at end.
- FINAL_IMAGE_PROMPT: Must specify composition, lighting, background, focal subject,
  brand colors, style cues, and include a "negative prompt" line (avoid clutter, watermarks, distorted hands, gibberish text).

Return EXACTLY:
A) COPY:
...

B) FINAL_IMAGE_PROMPT:
...
Extra notes: {extra}
     """.strip())
])

# We will call IMAGE model with this wrapper to force actual image generation
def wrap_image_request(final_image_prompt: str) -> str:
    return f"""
GENERATE AN IMAGE NOW. Do not answer with text-only.
Use this prompt exactly:

{final_image_prompt}
""".strip()


# ---------------- Gemini callers ----------------
def gemini_text_call(inputs: Dict[str, Any]) -> Dict[str, Any]:
    client = genai.Client(api_key=inputs["gemini_api_key"])
    model = inputs["text_model"]

    # format pro prompt messages -> single text
    messages = inputs["messages"]
    prompt_text = "\n".join([f"{m.type.upper()}: {m.content}" for m in messages])

    resp = client.models.generate_content(
        model=model,
        contents=prompt_text,
        config=types.GenerateContentConfig(response_modalities=["TEXT"]),
    )

    raw_text = extract_text(resp)
    copy, final_image_prompt = parse_copy_and_imageprompt(raw_text)

    return {
        **inputs,
        "copy": copy,
        "final_image_prompt": final_image_prompt,
        "text_raw": raw_text,
        "text_prompt_sent": prompt_text,
    }


def gemini_image_call(inputs: Dict[str, Any]) -> Dict[str, Any]:
    client = genai.Client(api_key=inputs["gemini_api_key"])
    model = inputs["image_model"]

    req = wrap_image_request(inputs["final_image_prompt"])

    resp = client.models.generate_content(
        model=model,
        contents=req,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=inputs["aspect_ratio"]),
        ),
    )

    img_bytes = extract_first_image_bytes(resp)

    return {
        "copy": inputs["copy"],
        "final_image_prompt": inputs["final_image_prompt"],
        "image_bytes": img_bytes,
        "image_raw": resp,  # for debug
        "text_raw": inputs["text_raw"],
        "text_prompt_sent": inputs["text_prompt_sent"],
    }


# ---------------- Chain (1-click, 2-step internal) ----------------
chain = (
    RunnablePassthrough.assign(
        brand_name=lambda x: x.get("brand_name") or "None",
        brand_colors=lambda x: x.get("brand_colors") or "None",
        extra=lambda x: x.get("extra") or "(none)",
        cta=lambda x: "Yes" if x.get("cta", True) else "No",
        hashtags=lambda x: "Yes" if x.get("hashtags", True) else "No",
    )
    | RunnableLambda(lambda x: {**x, "messages": TEXT_PROMPT.format_messages(**x)})
    | RunnableLambda(gemini_text_call)
    | RunnableLambda(gemini_image_call)
)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("🔑 Gemini API Key")
    gemini_key = st.text_input("API Key", type="password", placeholder="AIza...")

    st.divider()
    st.subheader("🧠 Models")
    # Text model: standard text LLM
    text_model = st.selectbox("Text model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
    # Image model: image-capable
    image_model = st.selectbox("Image model", ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"], index=0)

    st.caption("Si antes te devolvía solo texto, este flujo fuerza una llamada de **solo imagen**.")


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
    if not gemini_key:
        st.error("Introduce tu Gemini API Key en la barra lateral.")
        st.stop()
    if not simple_prompt.strip():
        st.error("Escribe un prompt simple.")
        st.stop()

    with st.spinner("Generando (texto → prompt pro → imagen)..."):
        result = chain.invoke({
            "gemini_api_key": gemini_key,
            "text_model": text_model,
            "image_model": image_model,
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

    with out.container():
        # Image
        img_bytes = result.get("image_bytes")
        if img_bytes:
            st.subheader("🖼️ Imagen")
            img = Image.open(BytesIO(img_bytes))
            st.image(img, use_container_width=True)
            st.download_button("Download image", img_bytes, "creative.png", "image/png", use_container_width=True)
        else:
            st.warning(
                "Sigo sin recibir bytes de imagen. Esto suele indicar que tu API key/proyecto "
                "no tiene habilitada salida de imagen para ese modelo/endpoint."
            )
            with st.expander("Debug"):
                st.code(result.get("final_image_prompt", ""), language="text")
                st.write(result.get("image_raw"))

        # Copy
        st.subheader("✍️ Copy")
        st.markdown(result.get("copy", ""))

        with st.expander("🧾 Prompt pro de imagen usado"):
            st.code(result.get("final_image_prompt", ""), language="text")
