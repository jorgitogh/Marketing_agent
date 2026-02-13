from io import BytesIO
from typing import Optional, Tuple

import streamlit as st
from PIL import Image

from google import genai
from google.genai import types

# ---------------- UI ----------------
st.set_page_config(page_title="Marketing Creative Generator", page_icon="🧠")
st.title("🧠 Marketing Creative Generator (1-click)")

with st.sidebar:
    st.subheader("🔑 Gemini API Key")
    gemini_key = st.text_input("API Key", type="password", placeholder="AIza...")
    st.caption("Usa Google AI Studio / Gemini API key.")
    st.divider()

# ---------------- Inputs ----------------
col1, col2, col3 = st.columns(3)

with col1:
    platform = st.selectbox("Platform", ["Instagram", "Facebook", "LinkedIn", "Blog", "E-mail"])
    aspect_ratio = st.selectbox("Aspect ratio", ["1:1", "4:5", "9:16", "16:9", "1.91:1"], index=1)

with col2:
    audience = st.selectbox("Audience", ["All", "Young adults", "Families", "Seniors", "Teenagers"])
    tone = st.selectbox("Tone", ["Normal", "Informative", "Inspiring", "Urgent", "Informal"])

with col3:
    style = st.selectbox(
        "Visual style",
        ["Photorealistic", "Minimalist", "3D clean render", "Flat illustration", "Modern collage", "Product mockup"]
    )
    model_choice = st.selectbox(
        "Gemini model",
        ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"]
    )

simple_prompt = st.text_area(
    "Simple prompt (what you want to promote)",
    placeholder="Ej: Promocionar una batidora potente para smoothies. Estética premium, fondo limpio, enfoque en el producto.",
    height=90
)

extra = st.text_area(
    "Extra notes (optional)",
    placeholder="Ej: incluir fruta fresca, luz natural suave, espacio para texto arriba, evitar colores chillones.",
    height=90
)

brand = st.text_input("Brand name (optional)", placeholder="Ej: Cecotec / YourBrand")
colors = st.text_input("Brand colors (optional)", placeholder="Ej: #111827, #8B5CF6, blanco")
cta = st.checkbox("Include CTA in the copy", value=True)
hashtags = st.checkbox("Add hashtags (for IG/FB)", value=True)

st.markdown("---")

# ---------------- Prompt builder ----------------
def build_pro_brief(
    simple_prompt: str,
    platform: str,
    aspect_ratio: str,
    audience: str,
    tone: str,
    style: str,
    brand: str,
    colors: str,
    extra: str,
    cta: bool,
    hashtags: bool,
) -> str:
    # Instrucciones para que Gemini devuelva texto + imagen y el texto venga estructurado
    return f"""
You are a senior marketing creative director and designer.

TASK
Create a complete marketing creative for {platform}:
1) A high-end marketing IMAGE (professional, brand-safe).
2) A ready-to-publish COPY in Spanish that matches the image.

INPUT (simple user intent)
{simple_prompt}

CONTEXT
- Platform: {platform}
- Aspect ratio: {aspect_ratio}
- Target audience: {audience}
- Tone: {tone}
- Visual style: {style}
- Brand name: {brand if brand else "None"}
- Brand colors: {colors if colors else "None"}
- CTA: {"Yes" if cta else "No"}
- Hashtags: {"Yes" if hashtags else "No"}

DESIGN RULES (image)
- Premium commercial look. Clean composition. Strong focal point.
- Avoid cheesy stock-photo vibe. Avoid clutter. Avoid watermarks.
- Ensure good lighting, clean background, and high readability.
- If you include text in the image, keep it minimal and perfectly spelled in Spanish.
- Leave some negative space for overlay if needed.

COPY RULES (text)
- Spanish.
- Adapt to {platform} norms.
- Keep it persuasive and clear.
- If CTA is enabled, include a clear CTA at the end.
- If hashtags enabled and platform is Instagram/Facebook, add 5–10 relevant hashtags at the end.
- Do NOT wrap output in quotes.

OUTPUT FORMAT (IMPORTANT)
Return:
A) COPY:
<final Spanish copy here>

B) IMAGE_DESCRIPTION:
<one concise sentence describing the generated image>

Do not add anything else.
Extra notes:
{extra if extra else "(none)"}
""".strip()

def extract_first_image_bytes(response) -> Optional[bytes]:
    if not response or not getattr(response, "candidates", None):
        return None
    for cand in response.candidates:
        content = getattr(cand, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for p in content.parts:
            inline = getattr(p, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return inline.data
    return None

def extract_text(response) -> str:
    # A veces viene en response.text, otras hay que recorrer parts
    txt = getattr(response, "text", None)
    if txt:
        return txt
    if not response or not getattr(response, "candidates", None):
        return ""
    for cand in response.candidates:
        content = getattr(cand, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for p in content.parts:
            t = getattr(p, "text", None)
            if t:
                return t
    return ""

def split_sections(text: str) -> Tuple[str, str]:
    # Parse muy simple por etiquetas A)/B)
    copy = text
    desc = ""
    if "A) COPY:" in text and "B) IMAGE_DESCRIPTION:" in text:
        part_a = text.split("A) COPY:", 1)[1]
        copy, desc = part_a.split("B) IMAGE_DESCRIPTION:", 1)
        copy = copy.strip()
        desc = desc.strip()
    return copy, desc

# ---------------- One-click generate ----------------
if st.button("🚀 Generate creative (image + copy)"):
    if not gemini_key:
        st.error("Mete tu Gemini API key en la barra lateral.")
        st.stop()
    if not simple_prompt.strip():
        st.error("Escribe un prompt simple (qué quieres promocionar).")
        st.stop()

    pro_prompt = build_pro_brief(
        simple_prompt=simple_prompt,
        platform=platform,
        aspect_ratio=aspect_ratio,
        audience=audience,
        tone=tone,
        style=style,
        brand=brand,
        colors=colors,
        extra=extra,
        cta=cta,
        hashtags=hashtags,
    )

    with st.expander("🧾 Pro prompt sent to Gemini", expanded=False):
        st.code(pro_prompt, language="text")

    try:
        client = genai.Client(api_key=gemini_key)

        # Pedimos MULTIMODAL: texto + imagen
        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio
            )
        )

        resp = client.models.generate_content(
            model=model_choice,
            contents=pro_prompt,
            config=config,
        )

        # extraer outputs
        img_bytes = extract_first_image_bytes(resp)
        txt = extract_text(resp)
        final_copy, img_desc = split_sections(txt)

        if img_bytes:
            image = Image.open(BytesIO(img_bytes))
            st.subheader("🖼️ Generated image")
            st.image(image, use_container_width=True)

            st.download_button(
                "Download image",
                data=img_bytes,
                file_name="creative.png",
                mime="image/png",
            )
        else:
            st.warning("No he recibido imagen (solo texto). Prueba el modelo Pro o revisa permisos/cuota.")
            st.write(resp)

        st.subheader("✍️ Generated copy")
        st.markdown(final_copy if final_copy else txt)

        if img_desc:
            st.caption(f"Image description: {img_desc}")

    except Exception as e:
        st.error(f"Gemini error: {e}")
