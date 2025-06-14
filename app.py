import os
import sys
import asyncio
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Add the project root to sys.path for module discovery
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the overall orchestration function
from main_orchestrator import run_overall_orchestration

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for all routes, allowing your frontend to communicate
CORS(app) 

# --- Health Check Route ---
@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "healthy", "message": "Flask backend is running!"}), 200

# --- Main Analysis Endpoint ---
@app.route('/analyze-agents', methods=['POST'])
async def analyze_agents_endpoint():
    """
    Receives pitch data from the frontend, triggers the multi-agent orchestration,
    and returns the combined analysis results.
    """
    try:
        data = request.json
        user_session_id = data.get('user_session_id')
        pitch_input_content = data.get('pitch_input_content')

        if not user_session_id or not pitch_input_content:
            return jsonify({
                "status": "failed",
                "error": "Missing user_session_id or pitch_input_content in request."
            }), 400

        print(f"\nAPI: Received request for session: {user_session_id}")

        # Correctly await the async function directly. Flask's async handler will manage the loop.
        orchestration_results = await run_overall_orchestration(
            user_session_id=user_session_id,
            pitch_content=pitch_input_content
        )

        # Log results for debugging
        print(f"API: Orchestration completed for session {user_session_id}. Status: {orchestration_results.get('status')}")

        return jsonify(orchestration_results), 200

    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"status": "failed", "error": str(e)}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Print a message to confirm backend startup
    print("Starting Flask Backend for OnlyFounders AI Agents...")
    print("Make sure your .env with GEMINI_API_KEY is in the root directory.")
    
    # Run the Flask app
    # Flask will automatically use an async-compatible server (like Werkzeug's async mode)
    # when an async view function is detected and flask[async] is installed.
    app.run(debug=True, host='0.0.0.0', port=5000)

    