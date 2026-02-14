# Marketing Agent

A Streamlit app that generates marketing creatives in one click:

- Spanish ad copy adapted to your platform and audience.
- A production-ready English image prompt.
- A generated image using Gemini image-capable models.

The flow is: `inputs -> text generation -> final image prompt -> image generation`.

## Features

- Platform-aware outputs for `Instagram`, `Facebook`, `LinkedIn`, `Blog`, and `E-mail`.
- Adjustable aspect ratios (`1:1`, `4:5`, `3:4`, `4:3`, `9:16`, `16:9`, `1.91:1`).
- Optional brand constraints (name and colors).
- Optional CTA and hashtag behavior for social channels.
- Download generated image as `creative.png`.
- Debug panel when image bytes are not returned.

## Tech Stack

- `Streamlit` for UI
- `LangChain Core` for prompt composition and runnable chain
- `google-genai` for Gemini text + image generation
- `Pillow` for image decoding/display

## Requirements

- Python `3.10+` recommended.
- A valid Gemini API key with image generation access enabled.

Install dependencies:

```bash
pip install -r requirements.txt
pip install pillow
```

## Run

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit and paste your Gemini API key in the sidebar.

## How To Use

1. Enter your Gemini API key.
2. Choose text and image models.
3. Fill campaign inputs (platform, audience, tone, style, prompt, optional brand rules).
4. Click `Generar (copy + imagen)`.
5. Review copy, inspect the generated image prompt, and download the image.

## Model Notes

Default model options in the UI:

- Text: `gemini-2.5-flash`, `gemini-2.5-pro`
- Image: `gemini-2.5-flash-image`, `gemini-3-pro-image-preview`

Use a model pair available in your project/region.

## Troubleshooting

- No image is returned:
  - Verify your key/project has image output enabled for the selected model.
  - Try the alternative image model in the dropdown.
  - Check the debug expander output in the app.
- API errors:
  - Confirm billing and permissions in your Google AI project.
- Dependency errors:
  - Reinstall with `pip install -r requirements.txt` and `pip install pillow`.

## Project Structure

- `app.py`: main Streamlit app and generation chain
- `requirements.txt`: Python dependencies
- `README.md`: project documentation
