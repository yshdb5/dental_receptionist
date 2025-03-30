# Mini Vocca Dental Assistant

As asked, here is a real-time **Vocal AI Receptionist** designed specifically for dental clinics.  
It uses advanced speech recognition, conversational AI, and calendar integration to interact with patients and help them manage their appointments.

---

## What can the assistant do?

- **Speak & Listen in Real-Time:**  
  The assistant understands and replies in **natural spoken French**, using a warm and professional tone.

- **Schedule Dental Appointments:**  
  It helps patients book, modify, or check the availability of appointments in the clinic’s **Google Calendar**.

- **Provide Information about the Clinic:**  
  It can instantly answer common questions about services, policies, or opening hours by searching in a built-in RAG model.

- **Engage in a Friendly Conversation:**  
  The assistant uses **GPT-4o** to understand context and communicate clearly, efficiently, and politely, like a real receptionist.

- **Live Video Avatar Animation:**  
  A simple animated avatar reacts while talking, giving a more engaging interaction for the patient.

---

## Use case example

When a patient joins the virtual reception room, the assistant will:
1. Greet the patient.
2. Ask for the type of dental care needed and the preferred date.
3. Check real-time availability via Google Calendar.
4. Propose a few free slots.
5. Collect the patient’s contact details.
6. Confirm the appointment.

It can also answer questions like:
> "Quels sont vos horaires d'ouverture ?"  
> "Quelle est votre politique d'annulation ?"  
> "Quels services proposez-vous ?"
> "Où se trouve votre clinique ?"
---

## Project structure
```
mini_vocca/
├── server/                           # Core application logic
│   ├── assets/                       # Avatar animation images
│   ├── cal/                          # Google Calendar integration
│   │   └── google_calendar.py
│   ├── chroma_db/                    # Chroma vector database for knowledge base
│   ├── processors/                   # Custom processors for appointments & info
│   │   ├── appointment.py
│   │   └── info.py
│   ├── bot-openai.py                 # Main pipeline: speech, LLM, TTS & processors
│   ├── runner.py                     # Pipeline runner setup
│   └── server.py                     # FastAPI server to expose bot endpoints
├── info.md                           # Knowledge base content
├── .env.example                      # Example environment variables
├── requirements.txt                  # Project dependencies
├── README.MD                         # Project documentation
```



## Environment Variables

Copy `env.example` to `.env` and configure:

```ini
# Required
DAILY_SAMPLE_ROOM_URL=https://yourdomain.daily.co/yourroom # (for joining the bot to the same room repeatedly for local dev)
DAILY_API_KEY=7df...
OPENAI_API_KEY=sk-PL...
CARTESIA_API_KEY=aeb...
BOT_IMPLEMENTATION=openai

# Optional Configuration
HOST=                    # Optional: Host address (defaults to 0.0.0.0)
FAST_API_PORT=           # Optional: Port number (defaults to 7860)
```

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
