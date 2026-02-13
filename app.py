import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from google import genai
from google.genai import types

load_dotenv()

# ---------------- UI ----------------
st.set_page_config(page_title="Content Generator 🤖", page_icon="🤖")
st.title("Content generator")

# ---------- helpers ----------
PLATFORM_DEFAULT_AR = {
    "Instagram": "4:5",
    "Facebook": "4:5",
    "LinkedIn": "1.91:1",
    "Blog": "16:9",
    "E-mail": "16:9",
}

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
    # Prompt tipo “brief” (suele funcionar MUY bien)
    text_rule = (
        "Include short, legible on-image text in Spanish (headline + optional CTA). Use clean typography."
        if wants_text_in_image else
        "Do NOT render any text in the image. Leave negative space for later overlay."
    )
    cta_rule = (
        "Make the design clearly support a call-to-action visually."
        if cta else
        "Do NOT emphasize CTA elements visually."
    )

    return f"""
You are a senior marketing designer. Create ONE premium marketing visual.

BRIEF
- Topic: {topic.strip() if topic else "N/A"}
- Platform: {platform}
- Target audience: {audience}
- Tone: {tone}
- Copy length context: {length}

STYLE
- Visual style: {image_style}
- Look: high-end, clean hierarchy, modern commercial, brand-safe, not cheesy stock-photo.

BRANDING
- Brand name: {brand_name.strip() if brand_name else "None"}
- Brand colors: {brand_colors.strip() if brand_colors else "None"}
- Logo instructions: {logo_instructions.strip() if logo_instructions else "No logo"}

RULES
- {text_rule}
- {cta_rule}
- Align visuals to these keywords if provided: {keywords.strip() if keywords else "None"}

MUST INCLUDE
{must_include.strip() if must_include else "- (none)"}

MUST AVOID (negative prompt)
{must_avoid.strip() if must_avoid else "- (none)"}
- Avoid watermarks, clutter, distorted hands, illegible gibberish text, low-res artifacts.

EXTRA DETAILS
{extra_image_details.strip() if extra_image_details else "(none)"}
""".strip()

def extract_first_image_bytes(response) -> bytes | None:
    """
    google-genai devuelve la imagen como inline_data en parts.
    Esto intenta encontrar el primer part con inline_data (image bytes).
    """
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

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("🔑 Groq API Key (texto)")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    st.divider()
    st.subheader("🖼️ Google Gemini API Key (imagen)")
    gemini_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")

    st.caption("Modelos: Nano Banana = gemini-2.5-flash-image | Pro = gemini-3-pro-image-preview")
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
st.subheader("🧩 Image generation (Gemini / Nano Banana)")

col1, col2 = st.columns(2)
with col1:
    image_style = st.selectbox(
        "Visual style:",
        ["Photorealistic", "Minimalist", "3D clean render", "Flat illustration", "Modern collage", "Product mockup"],
    )
    brand_name = st.text_input("Brand name (optional):", placeholder="e.g., YourBrand")
    brand_colors = st.text_input("Brand colors (optional):", placeholder="e.g., #111827, #8B5CF6, white")
with col2:
    model_choice = st.selectbox(
        "Gemini image model:",
        ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"]
    )
    aspect_ratio = st.selectbox(
        "Aspect ratio:",
        ["1:1", "3:4", "4:3", "9:16", "16:9", "1.91:1"],
        index=["1:1","3:4","4:3","9:16","16:9","1.91:1"].index(PLATFORM_DEFAULT_AR.get(platform, "1:1"))
    )
    wants_text_in_image = st.checkbox("Render text inside the image", value=False)

logo_instructions = st.text_input("Logo instructions (optional):", placeholder="e.g., place logo top-right, small")
must_include = st.text_area("Must include (optional):", placeholder="e.g., smartphone mockup, clean background, diverse people...")
must_avoid = st.text_area("Must avoid (optional):", placeholder="e.g., no medical needles, no neon colors, no clutter...")
extra_image_details = st.text_area(
    "Extra details for the image (your custom brief):",
    placeholder="e.g., premium, Scandinavian lighting, soft shadows, lots of whitespace..."
)

# ---------------- Buttons ----------------
colA, colB, colC = st.columns(3)
gen_text_btn = colA.button("✍️ Generate text")
gen_prompt_btn = colB.button("🧾 Generate image prompt")
gen_image_btn = colC.button("🖼️ Generate image (Gemini)")

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

# ---------------- Generate text (Groq) ----------------
if gen_text_btn:
    if not groq_key:
        st.error("Introduce tu Groq API Key en la barra lateral.")
        st.stop()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_key,
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
        st.error(f"Groq error: {e}")

# ---------------- Generate image prompt (no image yet) ----------------
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

if gen_prompt_btn:
    st.subheader("✅ Image prompt (ready)")
    st.code(img_prompt, language="text")

# ---------------- Generate image (Gemini) ----------------
if gen_image_btn:
    if not gemini_key:
        st.error("Introduce tu Gemini API Key en la barra lateral.")
        st.stop()

    try:
        client = genai.Client(api_key=gemini_key)

        # Config oficial para generación de imágenes (aspectRatio, numberOfImages, etc.) :contentReference[oaicite:2]{index=2}
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],  # solo imagen
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                number_of_images=1,
            ),
        )

        response = client.models.generate_content(
            model=model_choice,
            contents=img_prompt,
            config=config,
        )

        img_bytes = extract_first_image_bytes(response)
        if not img_bytes:
            st.warning("No he encontrado bytes de imagen en la respuesta. Muestra la respuesta para debug.")
            st.write(response)
            st.stop()

        image = Image.open(BytesIO(img_bytes))

        st.subheader("✅ Generated image")
        st.image(image, use_container_width=True)

        st.download_button(
            "Download image",
            data=img_bytes,
            file_name="marketing_image.png",
            mime="image/png",
        )

    except Exception as e:
        st.error(f"Gemini error: {e}")
