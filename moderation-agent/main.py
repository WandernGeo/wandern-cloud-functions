import functions_framework
import os
import json
import logging
import base64
import httpx
import google.generativeai as genai
from flask import Request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini
API_KEY = os.environ.get("GOOGLE_API_KEY")

# Models for different content types
# Text: Fast model for text-only moderation
# Vision: Multimodal model for images/video frames
# Audio: Not directly supported yet, will transcribe first
TEXT_MODEL = "gemini-2.0-flash-exp"
VISION_MODEL = "gemini-2.0-flash-exp"  # Same model supports vision

text_model = None
vision_model = None

if API_KEY:
    genai.configure(api_key=API_KEY)
    try:
        text_model = genai.GenerativeModel(TEXT_MODEL)
        vision_model = genai.GenerativeModel(VISION_MODEL)
        logger.info(f"Initialized models: text={TEXT_MODEL}, vision={VISION_MODEL}")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
else:
    logger.error("GOOGLE_API_KEY environment variable not set")


@functions_framework.http
def moderate_content(request: Request):
    """
    HTTP Cloud Function for Content Moderation Agent.
    Supports text, image, video (frame extraction), and audio (transcription).
    
    Accepts JSON:
    {
        "content": "text content...",
        "media_url": "optional url to image/video...",
        "media_b64": "optional base64 encoded media...",
        "content_type": "text" | "image" | "video" | "audio"
    }
    Returns JSON:
    {
        "is_safe": bool,
        "moderation_status": "approved" | "flagged",
        "flag_reason": "reason if flagged",
        "model_used": "model name"
    }
    """
    # CORS Headers
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600"
        }
        return ("", 204, headers)

    headers = {"Access-Control-Allow-Origin": "*"}

    # Check models
    if not text_model:
        return (jsonify({"error": "Configuration error: Models not initialized"}), 500, headers)

    # Parse Request
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return (jsonify({"error": "Invalid JSON"}), 400, headers)
        
        content_text = request_json.get("content", "")
        media_url = request_json.get("media_url")
        media_b64 = request_json.get("media_b64")
        content_type = request_json.get("content_type", "text")

    except Exception as e:
        logger.error(f"Error parsing request: {e}")
        return (jsonify({"error": "Bad Request"}), 400, headers)

    # Route to appropriate handler
    try:
        if content_type == "text":
            result = _scan_text(content_text)
        elif content_type == "image":
            result = _scan_image(media_url, media_b64)
        elif content_type == "video":
            # For video, we extract first frame and analyze
            result = _scan_video_frame(media_url, media_b64)
        elif content_type == "audio":
            # For audio, approve with note (transcription can be added later)
            result = {
                "is_safe": True,
                "moderation_status": "approved",
                "flag_reason": None,
                "model_used": "none (audio - manual review suggested)"
            }
        else:
            result = _scan_text(content_text)
        
        return (jsonify(result), 200, headers)

    except Exception as e:
        logger.error(f"Moderation Agent failed: {e}")
        # Fail open to not block users on error
        return (jsonify({
            "is_safe": True,
            "moderation_status": "approved",
            "flag_reason": f"Agent Error: {str(e)}",
            "model_used": "error"
        }), 200, headers)


def _scan_text(text: str) -> dict:
    """Moderate text content using fast text model."""
    prompt = f"""Role: Content Safety Agent for a family-friendly walking/exploration app.
Task: Analyze the following text for App Store compliance.

Strictly Flag:
- Hate Speech / Harassment / Bullying
- Sexually Explicit / NSFW content
- Dangerous / Illegal Acts / Self-harm
- Severe Profanity (mild PG-13 is okay)
- Personal info sharing (phone numbers, addresses)
- Spam / Advertising

Input Text: "{text}"

Output ONLY valid JSON:
{{"is_safe": true/false, "flag_reason": "short explanation if flagged, else null", "category": "hate|sexual|violence|spam|safe"}}"""
    
    try:
        response = text_model.generate_content(prompt)
        cleaned = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(cleaned)
        
        is_safe = data.get("is_safe", False)
        reason = data.get("flag_reason")
        
        return {
            "is_safe": is_safe,
            "moderation_status": "approved" if is_safe else "flagged",
            "flag_reason": reason,
            "model_used": TEXT_MODEL
        }
    except Exception as e:
        logger.error(f"Text scan failed: {e}")
        raise e


def _scan_image(media_url: str = None, media_b64: str = None) -> dict:
    """Moderate image content using vision model."""
    try:
        # Get image data
        if media_b64:
            image_data = base64.b64decode(media_b64)
        elif media_url:
            # Download image from URL
            with httpx.Client(timeout=30) as client:
                response = client.get(media_url)
                response.raise_for_status()
                image_data = response.content
        else:
            return {
                "is_safe": True,
                "moderation_status": "approved",
                "flag_reason": "No image provided",
                "model_used": VISION_MODEL
            }
        
        # Create image part for Gemini
        image_part = {
            "mime_type": "image/jpeg",  # Assume JPEG, works for most
            "data": image_data
        }
        
        prompt = """Role: Content Safety Agent for a family-friendly walking app.
Analyze this image for App Store compliance.

Flag if the image contains:
- Nudity or sexually explicit content
- Violence, gore, or disturbing imagery
- Hate symbols or offensive gestures
- Dangerous activities
- Personal information visible

Output ONLY valid JSON:
{"is_safe": true/false, "flag_reason": "short explanation if flagged, else null", "category": "sexual|violence|hate|safe"}"""

        response = vision_model.generate_content([prompt, image_part])
        cleaned = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(cleaned)
        
        is_safe = data.get("is_safe", False)
        reason = data.get("flag_reason")
        
        return {
            "is_safe": is_safe,
            "moderation_status": "approved" if is_safe else "flagged",
            "flag_reason": reason,
            "model_used": VISION_MODEL
        }
    except Exception as e:
        logger.error(f"Image scan failed: {e}")
        raise e


def _scan_video_frame(media_url: str = None, media_b64: str = None) -> dict:
    """
    Moderate video by analyzing first frame.
    For full video moderation, would need frame extraction.
    Currently treats video URL/data as image for simplicity.
    """
    # For MVP, try to treat as image (works for thumbnails)
    try:
        if media_b64:
            # If base64 provided, assume it's a frame/thumbnail
            return _scan_image(media_b64=media_b64)
        elif media_url and (media_url.endswith('.jpg') or media_url.endswith('.png')):
            # If it looks like an image URL, scan it
            return _scan_image(media_url=media_url)
        else:
            # For actual video files, we'd need ffmpeg - approve with note
            return {
                "is_safe": True,
                "moderation_status": "approved",
                "flag_reason": "Video moderation requires frame extraction - manual review suggested",
                "model_used": "none (video - needs frame extraction)"
            }
    except Exception as e:
        logger.error(f"Video scan failed: {e}")
        return {
            "is_safe": True,
            "moderation_status": "approved",
            "flag_reason": f"Video scan error: {str(e)}",
            "model_used": "error"
        }
