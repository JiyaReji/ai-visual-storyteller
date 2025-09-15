from flask import Flask, render_template, request, jsonify, send_file
import cohere
from diffusers import StableDiffusionPipeline
import requests
import torch
import spacy
from textblob import TextBlob
from PIL import Image
import time
import os
import uuid
import threading

app = Flask(__name__)

# Configure upload folder for saving generated files
UPLOAD_FOLDER = 'static/generated'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for API keys and models
COHERE_API_KEY = "x2YZ234aqWUBzfNSRcYSXbynVtcvh58h2j6vT8g2"  
ELEVENLABS_API_KEY = "sk_3d8a1291aefbba729a223b1e4624d5aaec2fbb2a66e8fe0a"  
VOICE_ID = "EXAVITQu4vr4xnSDxMaL" 

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global variable for the pipeline
pipe = None

# Initialize StableDiffusion pipeline on demand
def get_pipeline():
    global pipe
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            "dreamlike-art/dreamlike-anime-1.0", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
    return pipe

def text_to_speech(text, session_id):
    """Convert text to speech using ElevenLabs API."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_speech.mp3")
        with open(filename, "wb") as f:
            f.write(response.content)
        return filename
    else:
        print("Error with ElevenLabs API:", response.text)
        return None

def generate_story_cohere(prompt, model="command", max_tokens=1024, temperature=0.7):
    """Generate a longer story and retry if necessary."""
    try:
        response = co.generate(model=model, prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        story = response.generations[0].text.strip()

        # If story is too short, generate more content
        if len(story.split()) < 100:
            additional_response = co.generate(
                model=model, 
                prompt=f"Continue the story: {story}", 
                max_tokens=512, 
                temperature=temperature
            )
            story += " " + additional_response.generations[0].text.strip()

        return story
    except Exception as e:
        return f"Error: {str(e)}"

def split_story_into_scenes(story):
    """Split the story into meaningful scenes."""
    scenes = [scene.strip() for scene in story.split("\n\n") if scene.strip()]
    if len(scenes) < 3:
        # Try splitting by single newlines
        scenes = [scene.strip() for scene in story.split("\n") if scene.strip()]
        
    # If we still have too few scenes, split by sentences
    if len(scenes) < 3:
        import re
        scenes = [s.strip() for s in re.split(r'(?<=[.!?]) +', story) if s.strip()]
        
    return scenes

def extract_scene_details(scene_text):
    """Extract characters, locations, and objects from the scene."""
    doc = nlp(scene_text)
    characters = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC", "FAC"]]
    
    # Extract objects (direct and prepositional objects)
    objects = []
    for token in doc:
        if token.dep_ in ["dobj", "pobj"] and token.pos_ in ["NOUN", "PROPN"]:
            objects.append(token.text)
    
    return {
        "characters": ", ".join(set(characters)) if characters else "a cute little character",
        "locations": ", ".join(set(locations)) if locations else "a magical fairyland",
        "objects": ", ".join(set(objects[:5])) if objects else "some magical items",
    }

def analyze_sentiment(scene_text):
    """Determine the mood of the scene."""
    sentiment_score = TextBlob(scene_text).sentiment.polarity
    if sentiment_score > 0.3:
        return "happy, bright, cheerful"
    elif sentiment_score < -0.3:
        return "sad, emotional, warm and comforting"
    else:
        return "neutral, calm, relaxed"

def generate_image(scene_text, session_id, scene_number):
    """Generate an image for the scene using Stable Diffusion."""
    details = extract_scene_details(scene_text)
    mood = analyze_sentiment(scene_text)
    
    animated_prompt = (
        f"Child-friendly animated scene. {details['characters']} in {details['locations']} "
        f"with {details['objects']}. The scene feels {mood}. Cartoon-like, soft colors."
    )
    
    # Ensure prompt is within token limit
    animated_prompt = " ".join(animated_prompt.split()[:75])
    
    try:
        # Get or initialize the pipeline
        pipeline = get_pipeline()
        image = pipeline(animated_prompt, num_inference_steps=30).images[0]
        filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_scene_{scene_number}.png")
        image.save(filename)
        return filename
    except Exception as e:
        print(f"Image generation failed: {str(e)}")
        return None

def process_story(topic, details, creativity, session_id):
    """Process the entire story generation workflow."""
    results = {
        'status': 'processing',
        'message': 'Starting story generation...',
        'scenes': []
    }
    
    # Set temperature based on creativity level
    if creativity == "1":
        temperature = 0.3  # Kids Mode
    elif creativity == "2":
        temperature = 0.6  # Teens Mode
    else:
        temperature = 0.9  # Adults Mode
    
    # Generate the story
    full_prompt = f"Write a cute, engaging short story for children about {topic}. {details}"
    results['message'] = 'Generating story text...'
    story = generate_story_cohere(full_prompt, max_tokens=1024, temperature=temperature)

    if story.startswith("Error:"):
        results['status'] = 'error'
        results['message'] = story
        return results
    
    results['story'] = story
    results['message'] = 'Story generated. Splitting into scenes...'
    
    # Split into scenes
    scenes = split_story_into_scenes(story)
    
    # Select representative scenes for illustration
    if len(scenes) < 5:
        selected_scenes = scenes
    else:
        selected_scenes = [
            scenes[0],
            scenes[len(scenes) // 4],
            scenes[len(scenes) // 2],
            scenes[(3 * len(scenes)) // 4],
            scenes[-1]
        ]
    
    # Process each scene for images only
    for i, scene in enumerate(selected_scenes, start=1):
        results['message'] = f'Processing scene {i} of {len(selected_scenes)}...'
        
        scene_result = {
            'text': scene,
            'status': 'processing'
        }
        
        # Generate image
        results['message'] = f'Generating image for scene {i}...'
        image_path = generate_image(scene, session_id, i)
        
        if image_path:
            scene_result['image'] = image_path.replace('static/', '')
            scene_result['status'] = 'image_done'
        
        results['scenes'].append(scene_result)
    
    # Generate a single audio file for the entire story
    results['message'] = 'Generating audio for the story...'
    audio_path = text_to_speech(story, session_id)
    
    if audio_path:
        results['audio'] = audio_path.replace('static/', '')
    
    results['status'] = 'completed'
    results['message'] = 'Story generation completed!'
    return results

@app.route('/')
def index():
    return render_template('gowri.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.form
    topic = data.get('topic', '')
    details = data.get('details', '')
    creativity = data.get('creativity', '2')
    
    if not topic:
        return jsonify({'status': 'error', 'message': 'Topic cannot be empty'})
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    # Start processing in a background thread
    processing_thread = threading.Thread(
        target=process_story_background,
        args=(topic, details, creativity, session_id)
    )
    processing_thread.start()
    
    return jsonify({
        'status': 'started',
        'session_id': session_id,
        'message': 'Story generation has started!'
    })

def process_story_background(topic, details, creativity, session_id):
    """Background thread processing function."""
    results = process_story(topic, details, creativity, session_id)
    # Store results for later retrieval
    with open(os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_results.json"), 'w') as f:
        import json
        json.dump(results, f)

@app.route('/status/<session_id>', methods=['GET'])
def status(session_id):
    """Check status of a processing job."""
    result_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_results.json")
    
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            import json
            return jsonify(json.load(f))
    else:
        return jsonify({
            'status': 'processing',
            'message': 'Story is still being generated...'
        })

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_file(f'static/{filename}')

if __name__ == '__main__':
    app.run(debug=True) 