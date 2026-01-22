# 07 - Vision and OCR

Use LLMs to understand images, extract text, and analyze visual content.

## Why Vision?

Many financial documents exist as images or PDFs:
- Scanned earnings reports
- Screenshots of financial charts
- Tables in presentations
- Handwritten notes

Vision-enabled LLMs can extract structured data from these sources.

## What You'll Learn

- Sending images to vision-capable models
- Encoding images as base64
- Extracting text and data from images
- Analyzing charts and visualizations

## Setup

```bash
cd 07_vision_ocr
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Examples

### Describe an Image
```bash
python describe_image.py
```

### Extract Table Data
```bash
python extract_table.py
```

### Analyze a Chart
```bash
python read_chart.py
```

## Key Concepts

### Sending Images

Images can be sent as URLs or base64-encoded data:

```python
# URL (for publicly accessible images)
{"type": "image_url", "image_url": {"url": "https://..."}}

# Base64 (for local files)
{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_string}"}}
```

### Message Format with Images

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ]
    }
]
```

### Vision-Capable Models

- OpenAI: `gpt-4o`, `gpt-4o-mini`
- Google: `gemini-2.0-flash`, `gemini-2.5-pro`
- Anthropic: `claude-3-5-sonnet` (via OpenRouter)

## Sample Images

The `sample_images/` directory contains test images. Since we can't include actual images in this example, the scripts will generate simple test images programmatically using matplotlib.

## Finance Applications

- Extract data from earnings report tables
- Analyze stock chart patterns
- OCR historical financial documents
- Read data from trading platform screenshots

## Try It

1. Add your own financial chart and analyze it
2. Try extracting a table from a PDF screenshot
3. Compare OCR quality across different models
4. Chain vision with structured outputs to get typed data
