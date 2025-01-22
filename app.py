import json
import time
import os
import base64
import requests
import uuid
from flask import Flask, request, jsonify, Response
from cloudinary.uploader import upload as cloudinary_upload
from cloudinary.utils import cloudinary_url
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS for all origins
load_dotenv()

# Configuration
api_key = os.getenv("OPENAI_API_KEY")
cloudinary_cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
cloudinary_api_key = os.getenv("CLOUDINARY_API_KEY")
cloudinary_api_secret = os.getenv("CLOUDINARY_API_SECRET")
PROCTORING_API_BASE_URL = os.getenv("PROCTORING_API_BASE_URL", "http://127.0.0.1:8000")

def generate_unique_filename():
    """Generate a unique filename using UUID"""
    return str(uuid.uuid4())[:12]

def send_flag_notification(proctoring_id, flag_type, screenshot_url, pc_capture_url):
    """Send flag notification to the proctoring API"""
    url = f"{PROCTORING_API_BASE_URL}/assessments/proctoring-reports/{proctoring_id}/"

    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    payload = {
        "flags": [
            {
                "time": current_time,
                "flag": flag_type,
                "screenshot": screenshot_url,
                "pc_capture": pc_capture_url
            }
        ]
    }

    try:
        response = requests.patch(url, json=payload)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error sending flag notification: {str(e)}")
        return False

def check_and_send_flags(analysis, proctoring_id, screenshot_url, pc_capture_url):
    """Check analysis results and send notifications for any flags"""
    flag_mappings = {
        "multiple_faces": "multiple_faces_detected",
        "no_face": "no_face_detected",
        "face_partially_visible": "face_partially_visible",
        "looking_away": "looking_away",
        "suspicious_movements": "suspicious_movements"
    }

    for analysis_key, flag_type in flag_mappings.items():
        if analysis.get(analysis_key) == True:
            send_flag_notification(proctoring_id, flag_type, screenshot_url, pc_capture_url)

    prohibited_objects = analysis.get("prohibited_objects", {})
    if prohibited_objects.get("detected") == True:
        for item in prohibited_objects.get("items", []):
            flag_type = f"prohibited_object_{item}"
            send_flag_notification(proctoring_id, flag_type, screenshot_url, pc_capture_url)

@app.route('/process-images', methods=['POST'])
def process_images():
    data = request.json
    proctoring_id = data.get("proctoring_id")
    image_pairs = data.get("images")

    if not proctoring_id:
        return jsonify({"error": "proctoring_id is required"}), 400

    if not image_pairs or not isinstance(image_pairs, list):
        return jsonify({"error": "No images provided or invalid format"}), 400

    try:
        analysis_urls = []  # URLs for screenshots to be analyzed
        screenshot_urls = []  # Cloudinary URLs for screenshots
        pc_capture_urls = []  # Cloudinary URLs for PC captures

        for index, image_pair in enumerate(image_pairs):
            screenshot = image_pair.get("screenshot")
            pc_capture = image_pair.get("pc_capture")

            if not screenshot or not pc_capture:
                return jsonify({"error": f"Missing screenshot or pc_capture for image pair {index + 1}"}), 400

            try:
                start_time = time.time()

                # Upload screenshot
                screenshot_filename = f"{proctoring_id}_screen_{generate_unique_filename()}"
                if isinstance(screenshot, str) and 'base64,' in screenshot:
                    screenshot_data = base64.b64decode(screenshot.split('base64,')[1])
                else:
                    screenshot_data = base64.b64decode(screenshot)

                screenshot_result = cloudinary_upload(
                    screenshot_data,
                    folder="exam_proctoring",
                    public_id=screenshot_filename,
                    resource_type="image"
                )

                # Upload PC capture
                pc_capture_filename = f"{proctoring_id}_pc_{generate_unique_filename()}"
                if isinstance(pc_capture, str) and 'base64,' in pc_capture:
                    pc_capture_data = base64.b64decode(pc_capture.split('base64,')[1])
                else:
                    pc_capture_data = base64.b64decode(pc_capture)

                pc_capture_result = cloudinary_upload(
                    pc_capture_data,
                    folder="exam_proctoring",
                    public_id=pc_capture_filename,
                    resource_type="image"
                )

                end_time = time.time()
                print(f"Time taken to upload image pair {index + 1}: {end_time - start_time} seconds")

                # Store URLs
                screenshot_url = screenshot_result.get("secure_url")
                pc_capture_url = pc_capture_result.get("secure_url")

                analysis_urls.append(screenshot_url)  # Only screenshots are analyzed
                screenshot_urls.append(screenshot_url)
                pc_capture_urls.append(pc_capture_url)

            except Exception as e:
                return jsonify({"error": f"Failed to upload image pair {index + 1}: {str(e)}"}), 500

        # Prepare prompt for OpenAI API
        prompt_text = """You are an AI assistant helping with remote exam proctoring. Analyze the following image URLs and provide a JSON response for each image with the following structure. Ensure your  response should ONLY be in a single and valid JSON format is only the json and no additional text or prefix and you MUST NOT wrap it within JSON md markers such as this ('```json):
        [
        {
        "image_number": 1,
        "analysis": {
            "multiple_faces": false,  // true if more than one face is detected
            "no_face": false,  // true if no faces are detected
            "face_partially_visible": false,  // true if face is cut off or partially out of frame
            "looking_away": false,  // true if candidate is not looking at the screen
            "prohibited_objects": {
            "detected": false,  // true if any prohibited items are detected
            "items": []  // list of detected items: "phone", "book", "paper", "second_screen", etc.
            },
            "suspicious_movements": false,  // true if candidate shows suspicious head/eye movements
            "confidence_score_percentage": 95  // confidence in the analysis (0 to 100%)
        }
        }
        ]"""

        client = OpenAI(api_key=api_key)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url
                        }
                    }
                    for url in analysis_urls
                ]
            }
        ]

        start_time = time.time()

        response_data = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,
        )

        print(f"Time taken for OpenAI API response: {time.time() - start_time} seconds")

        content = response_data.choices[0].message.content

        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError as decode_error:
            return jsonify({"error": f"Failed to parse OpenAI response: {str(decode_error)}"}), 500

        final_response = []
        for i in range(len(analysis_urls)):
            analysis_result = {
                "image_url": analysis_urls[i],
                "analysis": parsed_content[i]["analysis"]
            }
            final_response.append(analysis_result)

            # Send flags with Cloudinary URLs
            check_and_send_flags(
                parsed_content[i]["analysis"],
                proctoring_id,
                screenshot_urls[i],
                pc_capture_urls[i]
            )

        return jsonify(final_response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     port = int(os.getenv("PORT", 5100))
#     app.run(host='0.0.0.0', port=port, debug=True)