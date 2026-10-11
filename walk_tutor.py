import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import random
from collections import deque

import numpy as np
import requests
import sounddevice as sd
import pyttsx3
from faster_whisper import WhisperModel

MODEL = "gemma3:1b"  # change this to the exact name from `ollama list`
URL = "http://localhost:11434/api/chat"
SAMPLE_RATE = 16000
BLOCK = int(SAMPLE_RATE * 0.1)      # 0.1 second blocks
SILENCE_SECONDS = 3.0               # a pause this long means "I'm done talking"
MAX_SECONDS = 180                   # safety limit for one answer (3 minutes)

# (problem, secret key idea, keywords that show the student has the key idea)
PROBLEMS = [
    ("Two Sum: given an array and a target, find two numbers that add up to the target.",
     "Use a hash map from value to index; for each x look up target - x. O(n).",
     ["hash", "dictionary", "map", "complement"]),
    ("Valid Parentheses: check if a string of brackets is balanced.",
     "Use a stack. Push opening brackets; on a closing bracket the stack top must be its match. Stack must be empty at the end.",
     ["stack"]),
    ("Binary Search: find a value in a sorted array in O(log n).",
     "Keep lo and hi; compare the middle element and discard half each step.",
     ["middle", "mid", "half"]),
    ("Reverse a linked list.",
     "Walk the list with prev, curr, next pointers and flip each node's link.",
     ["previous", "prev", "pointer"]),
    ("Find the longest substring without repeating characters.",
     "Sliding window with a set or map of last-seen positions; move the left edge when a repeat appears.",
     ["window", "two pointer", "set", "hash", "map"]),
]

SYSTEM_TEMPLATE = """You are a strict but friendly DSA coach. Your replies are spoken aloud.
PROBLEM: {problem}
SECRET KEY IDEA (never reveal it directly): {idea}

Never give the full solution or code.
Reply in at most 3 short sentences, plain words, no bullet points or symbols.
Each student message ends with a SYSTEM VERDICT. Always follow the verdict exactly."""

VERDICT_GOOD = (
    "SYSTEM VERDICT: the student mentioned the key idea. Confirm briefly in one sentence, "
    "then ask ONE follow-up question about time complexity or an edge case."
)
VERDICT_BAD = (
    "SYSTEM VERDICT: the student did NOT mention the key idea, so the answer is incomplete or wrong. "
    "Do NOT praise it. Start with 'Not quite yet', then give ONE small hint without revealing the answer."
)

print("Loading speech model...")
stt = WhisperModel("base.en", compute_type="int8")
tts = pyttsx3.init()
tts.setProperty("rate", 165)


def speak(text):
    print("\nCoach:", text, "\n")
    tts.say(text)
    tts.runAndWait()


def rms(block):
    return float(np.sqrt(np.mean(block ** 2)))


def calibrate(stream):
    """Listen to 1 second of background noise to set the 'speaking' threshold."""
    levels = [rms(stream.read(BLOCK)[0]) for _ in range(10)]
    threshold = max(0.008, float(np.mean(levels)) * 2.5)
    print(f"Background noise calibrated. Speaking threshold: {threshold:.4f}")
    return threshold


def listen(stream, threshold):
    """Wait for speech, record until a long pause, return the transcribed text."""
    print("Listening...")
    quiet_level = threshold * 0.6    # must drop clearly below this to count as a pause
    preroll = deque(maxlen=3)        # keep the moment just before speech starts
    chunks, started, quiet = [], False, 0
    silence_blocks = int(SILENCE_SECONDS / 0.1)
    max_blocks = int(MAX_SECONDS / 0.1)

    while True:
        data, _ = stream.read(BLOCK)
        level = rms(data)
        if not started:
            preroll.append(data)
            if level > threshold:
                started = True
                chunks.extend(preroll)
        else:
            chunks.append(data)
            quiet = quiet + 1 if level < quiet_level else 0
            if quiet >= silence_blocks or len(chunks) >= max_blocks:
                break

    audio = np.concatenate(chunks).flatten()
    segments, _ = stt.transcribe(audio, language="en")
    text = " ".join(s.text.strip() for s in segments).strip()
    print("You said:", text)
    return text


def ask(messages):
    # keep only the system prompt and the last few messages so the small model stays focused
    trimmed = messages[:1] + messages[1:][-6:]
    r = requests.post(
        URL,
        json={"model": MODEL, "messages": trimmed, "stream": False, "options": {"temperature": 0.3}},
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def new_problem():
    problem, idea, keywords = random.choice(PROBLEMS)
    messages = [{"role": "system", "content": SYSTEM_TEMPLATE.format(problem=problem, idea=idea)}]
    speak("Here is your problem. " + problem + " Explain your approach.")
    return problem, keywords, messages


problem, keywords, messages = new_problem()
print("Voice commands: 'next problem', 'repeat', 'quit'")

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK) as stream:
    threshold = calibrate(stream)
    while True:
        text = listen(stream, threshold)
        cleaned = text.lower().strip(" .!?,")
        if len(cleaned) < 3:
            continue                  # ignore noise and tiny fragments
        if cleaned == "quit":
            speak("Nice work. See you next walk.")
            break
        if cleaned == "repeat":
            speak(problem)
            continue
        if cleaned == "next problem":
            problem, keywords, messages = new_problem()
            continue

        # The code decides the verdict; the model only phrases the feedback.
        found = any(k in cleaned for k in keywords)
        print("[check] key idea found" if found else "[check] key idea NOT found")
        verdict = VERDICT_GOOD if found else VERDICT_BAD
        messages.append({"role": "user", "content": f'Student said: "{text}"\n{verdict}'})
        reply = ask(messages)
        messages.append({"role": "assistant", "content": reply})
        speak(reply)
