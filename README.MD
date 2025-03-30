# Mini Vocca Dental Assistant

This is a simple AI receptionist chatbot for dental clinics. 

It can make appointments, provide information about the clinic, and answer general questions about dental health.

# Simple Chatbot Server

A FastAPI server that manages bot instances and provides endpoints for both Daily Prebuilt and Pipecat client connections.

## Environment Variables

Copy `env.example` to `.env` and configure:

```ini
# Required API Keys
DAILY_SAMPLE_ROOM_URL=https://yourdomain.daily.co/yourroom # (for joining the bot to the same room repeatedly for local dev)
DAILY_API_KEY=7df...
OPENAI_API_KEY=sk-PL...
CARTESIA_API_KEY=aeb...
BOT_IMPLEMENTATION=openai

# Optional Configuration
HOST=                    # Optional: Host address (defaults to 0.0.0.0)
FAST_API_PORT=           # Optional: Port number (defaults to 7860)
```

## Available Bots

The server supports two bot implementations:

1. **OpenAI Bot** (Default)

   - Uses GPT-4 for conversation
   - Requires OPENAI_API_KEY

Select your preferred bot by setting `BOT_IMPLEMENTATION` in your `.env` file.

## Running the Server

Set up and activate your virtual environment:

```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the server:

```
python server.py
```
