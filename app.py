# app.py
# One-click Marketing Creative Generator (TEXT + IMAGE) using LangChain-style chain + Gemini (Nano Banana)
#
# Requirements:
#   pip install -U streamlit google-genai pillow langchain-core
#
# Run:
#   streamlit run app.py

from __future__ import annotations

from io import BytesIO
from typing import Optional, Dict, Any, Tuple

import streamlit as st
from PIL import Image

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from google import genai
from google.genai import types


# ---------------- UI ----------------
st.set_page_config(page_title="Marketing Creative Generator", page_icon="🧠", layout="wide")
st.title("🧠 Marketing Creative Generator (1-click)")
st.caption("Selecciona plataforma/estilo/público, escribe un prompt simple y genera **copy + imagen** con Gemini en un solo click.")


# ---------------- Helpers ----------------
PLATFORM_DEFAULT_AR = {
    "Instagram": "4:5",
    "Facebook": "4:5",
    "LinkedIn": "1.91:1",
    "Blog": "16:9",
    "E-mail": "16:9",
}

ASPECT_OPTIONS = ["1:1", "4:5", "3:4", "4:3", "9:16", "16:9", "1.91:1"]


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


def split_sections(text: str) -> Tuple[str, str]:
    copy, desc = text.strip(), ""
    if "A) COPY:" in text and "B) IMAGE_DESCRIPTION:" in text:
        part = text.split("A) COPY:", 1)[1]
        a, b = part.split("B) IMAGE_DESCRIPTION:", 1)
        copy, desc = a.strip(), b.strip()
    return copy, desc


# ---------------- LangChain Prompt (Pro Brief) ----------------
PRO_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior marketing creative director and designer. "
     "You produce premium, brand-safe creatives with clear hierarchy and professional aesthetics."),
    ("human",
     """
Create a complete marketing creative for {platform}.

INPUT (simple intent):
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

DESIGN RULES (image)
- Premium commercial look, clean composition, strong focal point.
- Avoid cheesy stock-photo vibe, clutter, watermarks, distorted hands/anatomy.
- If you include text in the image, keep it minimal and perfectly spelled in Spanish.
- Leave some negative space for overlays when appropriate.

COPY RULES (text)
- Spanish.
- Adapt to {platform} norms.
- Persuasive, clear, easy to read. No quotes.
- If CTA enabled, include a clear CTA at the end.
- If hashtags enabled and platform is Instagram/Facebook, include 5–10 relevant hashtags at the end.

EXTRA NOTES:
{extra}

OUTPUT FORMAT (IMPORTANT)
Return EXACTLY this format:

A) COPY:
<final Spanish copy here>

B) IMAGE_DESCRIPTION:
<one concise sentence describing the generated image>

IMPORTANT: You MUST generate the IMAGE (not only describe it) and also provide the COPY.
     """.strip()
    ),
])


# ---------------- Gemini Multimodal Call (Runnable) ----------------
def gemini_multimodal_call(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects:
      - gemini_api_key (str)
      - gemini_model (str)
      - aspect_ratio (str)
      - messages (list of LangChain BaseMessage)
    Returns:
      { "copy": str, "image_description": str, "image_bytes": Optional[bytes], "raw_text": str, "prompt_text": str }
    """
    api_key = inputs["gemini_api_key"]
    model = inputs.get("gemini_model", "gemini-2.5-flash-image")
    aspect_ratio = inputs.get("aspect_ratio", "4:5")
    messages = inputs["messages"]

    client = genai.Client(api_key=api_key)

    # Ask for both TEXT and IMAGE. (Some accounts/models may still return TEXT-only.)
    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
        ),
    )

    # Robust: concatenate prompt messages into a single text
    prompt_text = "\n".join([f"{m.type.upper()}: {m.content}" for m in messages])

    resp = client.models.generate_content(
        model=model,
        contents=prompt_text,
        config=config,
    )

    raw_text = extract_text(resp)
    copy, img_desc = split_sections(raw_text)
    img_bytes = extract_first_image_bytes(resp)

    return {
        "copy": copy,
        "image_description": img_desc,
        "image_bytes": img_bytes,
        "raw_text": raw_text,
        "prompt_text": prompt_text,
    }


# ---------------- Chain: inputs -> pro prompt -> gemini -> outputs ----------------
chain = (
    RunnablePassthrough.assign(
        brand_name=lambda x: x.get("brand_name") or "None",
        brand_colors=lambda x: x.get("brand_colors") or "None",
        extra=lambda x: x.get("extra") or "(none)",
        cta=lambda x: "Yes" if x.get("cta", True) else "No",
        hashtags=lambda x: "Yes" if x.get("hashtags", True) else "No",
    )
    | RunnableLambda(lambda x: {**x, "messages": PRO_PROMPT.format_messages(**x)})
    | RunnableLambda(gemini_multimodal_call)
)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("🔑 Gemini API Key")
    gemini_key = st.text_input("API Key", type="password", placeholder="AIza...")

    st.divider()
    st.subheader("⚙️ Model")
    model_choice = st.selectbox(
        "Gemini model",
        ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"],
        index=0
    )

    st.caption("Si no te devuelve imagen, prueba el modelo Pro.")


# ---------------- Front (form) ----------------
left, right = st.columns([1.05, 1])

with left:
    st.subheader("🧩 Inputs")

    platform = st.selectbox("Platform", ["Instagram", "Facebook", "LinkedIn", "Blog", "E-mail"])
    default_ar = PLATFORM_DEFAULT_AR.get(platform, "1:1")
    default_idx = ASPECT_OPTIONS.index(default_ar) if default_ar in ASPECT_OPTIONS else 0
    aspect_ratio = st.selectbox("Aspect ratio", ASPECT_OPTIONS, index=default_idx)

    col1, col2 = st.columns(2)
    with col1:
        audience = st.selectbox("Audience", ["All", "Young adults", "Families", "Seniors", "Teenagers"])
        tone = st.selectbox("Tone", ["Normal", "Informative", "Inspiring", "Urgent", "Informal"])
    with col2:
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
    output_placeholder = st.empty()


# ---------------- Action ----------------
if generate:
    if not gemini_key:
        st.error("Introduce tu Gemini API Key en la barra lateral.")
        st.stop()
    if not simple_prompt.strip():
        st.error("Escribe un prompt simple.")
        st.stop()

    with st.spinner("Generando creative con Gemini..."):
        result = chain.invoke({
            "gemini_api_key": gemini_key,
            "gemini_model": model_choice,
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

    # ---- Render outputs ----
    copy = result.get("copy") or result.get("raw_text") or ""
    img_desc = result.get("image_description") or ""
    img_bytes = result.get("image_bytes")

    with output_placeholder.container():
        if img_bytes:
            st.subheader("🖼️ Imagen")
            try:
                image = Image.open(BytesIO(img_bytes))
                st.image(image, use_container_width=True)
                st.download_button(
                    "Download image",
                    data=img_bytes,
                    file_name="creative.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"Recibí bytes pero no pude abrir la imagen: {e}")
        else:
            st.warning("No he recibido imagen (solo texto). Si te pasa, prueba el modelo Pro o revisa si tu cuenta tiene habilitada generación de imágenes.")
            # Debug mínimo: muestra estructura
            with st.expander("Debug (respuesta texto / prompt enviado)"):
                st.code(result.get("prompt_text", ""), language="text")
                st.write({"raw_text": result.get("raw_text", "")})

        st.subheader("✍️ Copy")
        st.markdown(copy)

        if img_desc:
            st.caption(f"Image description: {img_desc}")
