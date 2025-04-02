import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from pipecat.transcriptions.language import Language
from datetime import datetime

from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    OutputImageRawFrame,
    SpriteFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport, DailyTranscriptionSettings

from cal.google_calendar import GoogleCalendarService
from processors.appointment import AppointmentProcessor
from processors.info import InfoProcessor

load_dotenv(override=True)
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sprites = []
script_dir = os.path.dirname(__file__)

for i in range(1, 26):
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    with Image.open(full_path) as img:
        sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

flipped = sprites[::-1]
sprites.extend(flipped)

quiet_frame = sprites[0]
talking_frame = SpriteFrame(images=sprites)

calendar_service = GoogleCalendarService(
    credentials_file=os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"),
    calendar_id=os.getenv("GOOGLE_CALENDAR_ID")
)

class TalkingAnimation(FrameProcessor):
    """
    Processor to animate the bot's avatar while talking.
    """
    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True

        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame, direction)


async def main():
    async with (aiohttp.ClientSession() as session):
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Réceptionniste",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=576,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
                transcription_settings=DailyTranscriptionSettings(language=Language.FR_FR)
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="65b25c5d-ff07-4687-a04c-da2f43ef6fa9",
            params=CartesiaTTSService.InputParams(
                language=Language.FR
            )
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            temperature = 0.2
        )

        current_date = datetime.now().strftime("%d/%m/%Y")
        system_prompt = f"""
            Vous êtes Sophie, la réceptionniste virtuelle du cabinet dentaire "Tout Sourire".
            
            Vous vous exprimez uniquement et exclusivement en français.
            
            Nous sommes aujourd'hui le {current_date}.

            Votre style est chaleureux et professionnel, mais vous allez toujours à l’essentiel. Soyez claire, concise et naturelle, sans tourner autour du pot.

            Votre rôle est d’aider le patient à réserver un rendez-vous :
            - Demandez le type de soin souhaité
            - Demandez la date souhaitée
            - Utilisez checkAvailability() pour consulter les créneaux
            - Proposez les créneaux disponibles de façon simple, maximum 3 créneaux
            - Recueillez le nom, le téléphone et, si possible, l'email
            - Utilisez bookAppointment() pour confirmer le rendez-vous

            Rappels :
            - Politique d'annulation : prévenir 24h à l'avance
            - Horaires : lundi à vendredi, de 9h00 à 17h00.
            - Pour un horaire comme "9h00", dites "neuf heures". Pour un horaire comme "17h30", dites "dix-sept heures trente".
            - Si vous utilisez des heures ou des dates, utilisez TOUJOURS le format français : "9h00", "17h00", "1er avril". Ne jamais utiliser d'anglais.
            - Urgences : 01 23 45 67 89

            Très important : Ne dites jamais que le rendez-vous est confirmé avant d'avoir appelé bookAppointment().
        """

        messages = [{"role": "system", "content": system_prompt}]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # Create the appointment processor for handling appointment requests and bookings
        appointment_processor = AppointmentProcessor(calendar_service, context, context_aggregator)
        ta = TalkingAnimation()
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        info_processor = InfoProcessor(context, context_aggregator)

        # Register the processors' methods with the LLM
        llm.register_function("checkAvailability", appointment_processor.handle_check_availability)
        llm.register_function("bookAppointment", appointment_processor.handle_book_appointment)
        llm.register_function("getInfo", info_processor.handle_get_info)

        pipeline = Pipeline([
            transport.input(),
            rtvi,
            context_aggregator.user(),
            llm,
            appointment_processor,
            info_processor,
            tts,
            ta,
            transport.output(),
            context_aggregator.assistant(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )

        appointment_processor.set_task(task)
        info_processor.set_task(task)

        await task.queue_frame(quiet_frame)

        # Set the bot as ready when the client is ready
        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()

        # Capture the first participant's transcription
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        # Cancel the task when the participant leaves
        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"Participant left: {participant}")
            await task.cancel()

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())