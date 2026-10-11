# Walk Tutor

A voice DSA coach you use on a walk. It runs fully offline on a laptop.

## How it works
  - Speech to text: faster-whisper
  - Coach: Gemma 3 1B through Ollama
  - Speech out: pyttsx3
  - The code checks your answer for the key idea; the model only writes the feedback.

## Setup
  1. Install Ollama and run: ollama pull gemma3:1b
  2. pip install -r requirements.txt
  3. python walk_tutor.py

Voice commands: "next problem", "repeat", "quit".
