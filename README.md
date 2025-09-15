 OneSup: AI-Powered Story Generator & Illustrator

OneSup is a Flask-based web application that generates animated children's stories using AI, illustrates key scenes with beautiful images, and provides audio narration — all automatically!

 Features

-Story Generation: Creates engaging, age-tailored short stories from a simple topic and description.
-Scene Illustration: Uses AI image generation to produce cartoon-style images for key story moments.
-Audio Narration: Converts the story into a soothing voice-over.
-Web Interface: Clean HTML interface to input topic, select creativity levels, and view results.

Powered By

- Natural Language Generation (NLG)
- Named Entity Recognition (NER) with spaCy
- Sentiment Analysis with TextBlob
- Image Generation using Stable Diffusion
- Audio Generation using TTS APIs

Project Structure

onesup/
│
├── onesup.py # Main Flask application
├── templates/
│ └── gowri.html # Web interface (form and UI)
├── static/
│ └── generated/ # Auto-generated images and audio
└── README.md # Project description

