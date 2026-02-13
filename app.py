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

# Helpers
PLATFORM_ASPECT = {
    "Instagram": "1:1 (square) or 4:5 (feed portrait) / 9:16 (story)",
    "Facebook": "1:1 or 4:5 / 16:9 (banner)",
    "LinkedIn": "1.91:1 (landscape) or 4:5 (portrait)",
    "Blog": "16:9 (hero) or 3:2",
    "E-mail": "wide banner ~ 16:9 (safe center)",
}

def infer_aspect_ratio(platform: str) -> str:
    # “Default” simple (puedes ajustar a tu gusto)
    if platform == "LinkedIn":
        return "1.91:1"
    if platform == "Blog":
        return "16:9"
    if platform == "E-mail":
        return "16:9"
    if platform in ["Instagram", "Facebook"]:
        return "4:5"
    return "1:1"

def llm_generate(llm, prompt: str) -> str:
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a digital marketing expert specialized in SEO and persuasive copywriting."),
        ("human", "{prompt}"),
    ])
    chain = template | llm | StrOutputParser()
    return chain.invoke({"prompt": prompt})

def build_image_prompt(
    topic: str,
    platform: str,
    tone: str,
    audience: str,
    length: str,
    cta: bool,
    keywords: str,
    extra_image_details: str,
    image_style: str,
    brand_name: str,
    brand_colors: str,
    logo_instructions: str,
    must_include: str,
    must_avoid: str,
    wants_text_in_image: bool,
):
    aspect = infer_aspect_ratio(platform)
    platform_hint = PLATFORM_ASPECT.get(platform, aspect)

    # Reglas prácticas para outputs “marketing”
    text_rule = (
        "Include short, legible on-image text (headline + optional CTA) in Spanish, "
        "using clean typography and correct spelling. Keep text minimal and readable."
        if wants_text_in_image else
        "Do NOT render any text in the image. Leave clean negative space where text could be overlaid later."
    )

    cta_rule = (
        "Design should clearly support a call-to-action (CTA) visually (button-like area or focal element)."
        if cta else
        "Do NOT emphasize any CTA element visually."
    )

    # Prompt final (estructurado, estilo “brief”)
    prompt = f"""
You are a senior marketing designer. Create ONE high-end marketing visual.

BRIEF
- Topic / concept: {topic.strip() if topic else "N/A"}
- Platform: {platform} (typical formats: {platform_hint})
- Target audience: {audience}
- Tone: {tone}
- Copy length context (for layout density): {length}

FORMAT
- Aspect ratio: {aspect}
- Composition: strong focal point, clean hierarchy, premium commercial look, social-ready framing.

BRANDING
- Brand name (if any): {brand_name.strip() if brand_name else "None"}
- Brand colors (if any): {brand_colors.strip() if brand_colors else "None"}
- Logo instructions: {logo_instructions.strip() if logo_instructions else "No logo"}

CONTENT DIRECTION
- Style: {image_style}
- {text_rule}
- {cta_rule}
- If keywords are provided, align visuals to these concepts: {keywords.strip() if keywords else "None"}

MUST INCLUDE
{must_include.strip() if must_include else "- (none)"}

MUST AVOID (NEGATIVE PROMPT)
{must_avoid.strip() if must_avoid else "- (none)"}
- No watermarks, no fake brand marks, no illegible gibberish text, no distorted anatomy/hands, no cluttered layout.

EXTRA DETAILS
{extra_image_details.strip() if extra_image_details else "(none)"}

Output: generate the image accordingly.
""".strip()

    return prompt

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("🔑 Groq API Key (texto)")
    api_key = st.text_input("API Key", type="password", placeholder="gsk_...")
    st.caption("Consíguela en la consola de Groq.")
    st.divider()

    st.subheader("🖼️ Google Gemini API (imagen)")
    google_api_key = st.text_input("Google API Key", type="password", placeholder="AIza... (opcional)")
    st.caption("Para generar imágenes con Nano Banana vía Gemini API (si luego lo conectas).")
    st.divider()

# ---------------- Form (texto) ----------------
topic = st.text_input("Topic:", placeholder="e.g., nutrition, mental health, routine check-ups...")
platform = st.selectbox("Platform:", ['Instagram', 'Facebook', 'LinkedIn', 'Blog', 'E-mail'])
tone = st.selectbox("Message tone:", ['Normal', 'Informative', 'Inspiring', 'Urgent', 'Informal'])
length = st.selectbox("Text length:", ['Short', 'Medium', 'Long'])
audience = st.selectbox("Target audience:", ['All', 'Young adults', 'Families', 'Seniors', 'Teenagers'])
cta = st.checkbox("Include CTA")
hashtags = st.checkbox("Return Hashtags")
keywords = st.text_area("Keywords (SEO):", placeholder="Example: wellness, preventive healthcare...")

# ---------------- Form (imagen) ----------------
st.markdown("---")
st.subheader("🧩 Image Prompt (para Nano Banana / Gemini API)")

col1, col2 = st.columns(2)
with col1:
    image_style = st.selectbox(
        "Visual style:",
        ["Photorealistic", "Minimalist", "3D clean render", "Flat illustration", "Modern collage", "Product mockup"],
    )
    brand_name = st.text_input("Brand name (optional):", placeholder="e.g., Cecotec / YourBrand")
    brand_colors = st.text_input("Brand colors (optional):", placeholder="e.g., #111827, #8B5CF6, white")
with col2:
    wants_text_in_image = st.checkbox("Render text inside the image (harder, more 'Pro')", value=False)
    logo_instructions = st.text_input("Logo instructions (optional):", placeholder="e.g., place logo top-right, small")
    st.caption(f"Suggested aspect ratio for {platform}: **{infer_aspect_ratio(platform)}**")

must_include = st.text_area(
    "Must include (optional):",
    placeholder="e.g., a diverse group of people smiling; smartphone mockup; product centered; clean background...",
)
must_avoid = st.text_area(
    "Must avoid / negative prompt (optional):",
    placeholder="e.g., no medical imagery; no needles; no overly happy stock-photo vibe; no neon colors...",
)
extra_image_details = st.text_area(
    "Extra details for the image (your custom brief):",
    placeholder="e.g., make it feel premium, Scandinavian lighting, soft shadows, lots of whitespace, modern typography space...",
)

# ---------------- Action buttons ----------------
colA, colB = st.columns(2)

def build_text_prompt():
    return f"""
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

with colA:
    gen_text = st.button("✍️ Generate text")
with colB:
    gen_image_prompt = st.button("🖼️ Generate image prompt")

# ---------------- Run text generation ----------------
if gen_text:
    if not api_key:
        st.error("Introduce tu Groq API Key en la barra lateral para continuar.")
        st.stop()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    try:
        res = llm_generate(llm, build_text_prompt())
        st.subheader("✅ Generated copy")
        st.markdown(res)
    except Exception as e:
        st.error(f"Error: {e}")

# ---------------- Run image prompt generation (no API call, only prompt) ----------------
if gen_image_prompt:
    img_prompt = build_image_prompt(
        topic=topic,
        platform=platform,
        tone=tone,
        audience=audience,
        length=length,
        cta=cta,
        keywords=keywords,
        extra_image_details=extra_image_details,
        image_style=image_style,
        brand_name=brand_name,
        brand_colors=brand_colors,
        logo_instructions=logo_instructions,
        must_include=must_include,
        must_avoid=must_avoid,
        wants_text_in_image=wants_text_in_image,
    )

    st.subheader("✅ Image prompt (ready for Nano Banana)")
    st.code(img_prompt, language="text")
    st.caption("Tip: si activas texto dentro de la imagen, suele ir mejor con el modelo Pro.")
