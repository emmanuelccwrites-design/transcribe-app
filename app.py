import os
from flask import Flask, request, render_template, jsonify
import whisper
from werkzeug.utils import secure_filename

# Add FFmpeg to PATH specifically for this process
ffmpeg_path = r"C:\Users\bocal\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.abspath('uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the whisper model once on startup
print("Loading Whisper model (this may take a moment on first run)...")
# 'base' is a good balance between speed and accuracy for an MVP.
# It requires about 1GB of VRAM or can run fine on a decent CPU.
model = whisper.load_model("base", device="cpu")  # Load model on CPU for compatibility
print("Whisper model loaded!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file part in the request'}), 400
    
    file = request.files['audio_file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        # Save to a temporary location and verify
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Verify that the file exists and is not empty
        if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
            return jsonify({'error': 'Uploaded file is empty or could not be saved'}), 400
        
        try:
            # Directly transcribe the uploaded file using ffmpeg for conversion
            result = model.transcribe(filepath, fp16=False)
            text = result["text"]
            
            # Clean up the file after transcription
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Failed to remove temporary file {filepath}: {e}")
                
            return jsonify({'text': text})
            
        except Exception as e:
            # Capture detailed Whisper errors
            error_msg = str(e)
            if 'Failed to load audio' in error_msg:
                error_msg = 'Failed to load audio. Ensure ffmpeg is installed and the file is a valid audio format.'
            elif 'reshape' in error_msg.lower():
                error_msg = 'Audio processing failed (zero-length audio). Verify the uploaded file is a supported audio format.'
            elif 'empty' in error_msg.lower():
                error_msg = 'Uploaded file appears to be empty or invalid audio.'
            return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
