import functions_framework
import os
import json
import logging
import google.generativeai as genai
from flask import Request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Gemini
API_KEY = os.environ.get("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
    # Use Gemini 1.5 Flash for speed and cost/multimodal capabilities
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    logger.error("GOOGLE_API_KEY environment variable not set")
    model = None

@functions_framework.http
def moderate_content(request: Request):
    """
    HTTP Cloud Function for Content Moderation Agent.
    Accepts JSON:
    {
        "content": "text content...",
        "media_url": "optional url...",
        "content_type": "text" | "image" | "video" | "audio"
    }
    Returns JSON:
    {
        "is_safe": bool,
        "moderation_status": "approved" | "flagged",
        "flag_reason": "reason if flagged",
        "analysis_raw": {} 
    }
    """
    # 1. CORS Headers
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600"
        }
        return ("", 204, headers)

    headers = {"Access-Control-Allow-Origin": "*"}

    # 2. Key Check
    if not model:
        return (jsonify({"error": "Configuration error: API Key missing"}), 500, headers)

    # 3. Parse Request
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return (jsonify({"error": "Invalid JSON"}), 400, headers)
        
        content_text = request_json.get("content", "")
        # media_url = request_json.get("media_url") # TODO: Handle media download for vision
        content_type = request_json.get("content_type", "text")

    except Exception as e:
        logger.error(f"Error parsing request: {e}")
        return (jsonify({"error": "Bad Request"}), 400, headers)

    # 4. Agent Logic
    try:
        if content_type == "text":
            result = _scan_text(content_text)
        else:
            # Placeholder for media (requires downloading bytes or passing URI if supported)
            # For MVP, verify if we can pass image URLs to Gemini 1.5 Flash directly? 
            # Usually requires valid Part object with mime_type and data.
            # We will implement Text first, fail-safe Media.
            result = {
                "is_safe": True, 
                "moderation_status": "approved", 
                "flag_reason": "Media moderation pending implementation"
            }
        
        return (jsonify(result), 200, headers)

    except Exception as e:
        logger.error(f"Moderation Agent failed: {e}")
        # Fail open or closed? Let's fail open with warning for now to not block users on error
        return (jsonify({
            "is_safe": True, 
            "moderation_status": "approved", 
            "flag_reason": f"Agent Error: {str(e)}"
        }), 200, headers)

def _scan_text(text: str) -> dict:
    prompt = f"""
    Role: Content Safety Agent.
    Task: Analyze the following text for App Store compliance.
    
    Strictly Flag:
    - Hate Speech / Harassment
    - Sexually Explicit / NSFW
    - Dangerous / Illegal Acts
    - Severe Profanity (PG-13 is okay)

    Input Text: "{text}"

    Output JSON:
    {{
        "is_safe": true/false,
        "flag_reason": "short explanation if flagged, else null",
        "category": "hate|sexual|violence|safe"
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        cleaned = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(cleaned)
        
        is_safe = data.get("is_safe", False)
        reason = data.get("flag_reason")
        
        return {
            "is_safe": is_safe,
            "moderation_status": "approved" if is_safe else "flagged",
            "flag_reason": reason
        }
    except Exception as e:
        logger.error(f"Gemini scan failed: {e}")
        raise e
