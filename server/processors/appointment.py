import datetime
from typing import Dict, List, Any, Optional

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from cal.google_calendar import GoogleCalendarService
from pipecat.pipeline.task import PipelineTask

class AppointmentProcessor(FrameProcessor):
    """ Processor to handle appointment scheduling."""
    def __init__(self, calendar_service: GoogleCalendarService, llm_context: OpenAILLMContext, context_aggregator):
        super().__init__()
        self.calendar_service = calendar_service
        self.llm_context = llm_context
        self.appointment_state = {}
        self.context_aggregator = context_aggregator
        self.task: Optional[PipelineTask] = None

        # Set up LLM tools which are the functions that can be called from the LLM
        current_tools = getattr(self.llm_context, "tools", []) or []
        self.llm_context.set_tools(current_tools + [
            {
                "type": "function",
                "function": {
                    "name": "checkAvailability",
                    "description": "Vérifie les créneaux disponibles.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "serviceType": {"type": "string"}
                        },
                        "required": ["date", "serviceType"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "bookAppointment",
                    "description": "Réserve un rendez-vous",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "integer", "description": "Timestamp de début"},
                            "end": {"type": "integer", "description": "Timestamp de fin"},
                            "patientInfo": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "phone": {"type": "string"},
                                    "email": {"type": "string"}
                                },
                                "required": ["name", "phone"]
                            }
                        },
                        "required": ["start", "end", "patientInfo"]
                    }
                }
            }
        ])

    def set_task(self, task: PipelineTask):
        self.task = task

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def handle_check_availability(self, function_name, tool_call_id, args, llm, context, result_callback):
        date_str = args["date"]
        service_type = args["serviceType"]
        await self._handle_check_availability(date_str, service_type)
        await result_callback(None)
        await self._auto_continue()

    async def handle_book_appointment(self, function_name, tool_call_id, args, llm, context, result_callback):
        start = args["start"]
        end = args["end"]
        patient_info = args["patientInfo"]
        await self._handle_book_appointment(start, end, patient_info)
        await result_callback(None)
        await self._auto_continue()

    async def _auto_continue(self):
        if self.task:
            await self.task.queue_frames([self.context_aggregator.user().get_context_frame()])

    async def _handle_check_availability(self, date_str: str, service_type: str):
        slots = await self.process_appointment_request(date_str, service_type)
        if slots:
            slots_text = ""
            for slot in slots[:3]:
                start_ts = int(slot['start_datetime'].timestamp())
                end_ts = int(slot['end_datetime'].timestamp())
                slots_text += (
                    f"- De {slot['start']} à {slot['end']} (start: {start_ts}, end: {end_ts})\n"
                )
            response = (
                f"Voici les créneaux disponibles pour le {date_str} ({service_type}):\n"
                f"{slots_text.strip()}\n"
                f"Lequel souhaitez-vous ?"
            )
        else:
            response = f"Je suis désolée, aucun créneau disponible le {date_str}. Voulez-vous essayer une autre date ?"

        self.llm_context.add_message({"role": "system", "content": response})

    async def _handle_book_appointment(self, start: int, end: int, patient_info: Dict[str, str]):
        appointment = await self.book_appointment(start, end, patient_info)

        if appointment:
            response = (
                f"Parfait ! Rendez-vous confirmé pour {appointment['service']} "
                f"le {appointment['start']} au nom de {appointment['patient']}.\n"
                f"Vous recevrez une confirmation par email."
            )
        else:
            response = "Je suis désolée, je n'ai pas pu réserver ce créneau. Souhaitez-vous essayer un autre ?"

        self.llm_context.add_message({"role": "system", "content": response})

    async def process_appointment_request(self, date_str: str, service_type: str) -> List[Dict[str, Any]]:
        try:
            date = self._parse_date(date_str)
            duration = self._get_service_duration(service_type)

            available_slots = await self.calendar_service.get_available_slots(date, duration)
            self.appointment_state = {
                'date': date,
                'service_type': service_type,
                'available_slots': available_slots,
                'duration': duration
            }
            return available_slots

        except Exception as e:
            logger.error(f"Erreur lors de la recherche des créneaux : {e}")
            return []

    async def book_appointment(self, start: int, end: int, patient_info: Dict[str, str]) -> Optional[Dict[str, Any]]:
        try:
            service_type = self.appointment_state.get('service_type')
            slots = self.appointment_state.get('available_slots', [])

            if not service_type or not slots:
                logger.error("État de rendez-vous incomplet. Impossible de réserver.")
                return None

            # Matching souple : tolérance de 60 secondes
            selected_slot = next(
                (slot for slot in slots
                 if abs(int(slot['start_datetime'].timestamp()) - start) < 60
                 and abs(int(slot['end_datetime'].timestamp()) - end) < 60),
                None
            )

            if not selected_slot:
                logger.error(f"Créneau introuvable pour {start}-{end}.")
                return None

            appointment = await self.calendar_service.create_appointment(
                start_time=selected_slot['start_datetime'],
                end_time=selected_slot['end_datetime'],
                patient_name=patient_info.get('name', ''),
                patient_email=patient_info.get('email'),
                patient_phone=patient_info.get('phone'),
                service_type=service_type
            )

            if appointment:
                return {
                    'start': appointment['start'],
                    'end': appointment['end'],
                    'patient': appointment['patient'],
                    'service': appointment['service']
                }

            return None

        except Exception as e:
            logger.error(f"Erreur lors de la réservation : {e}")
            return None

    def _parse_date(self, date_str: str) -> datetime.date:
        for fmt in ('%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d %B %Y'):
            try:
                return datetime.datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Format de date invalide: {date_str}")

    def _get_service_duration(self, service_type: str) -> int:
        durations = {
            "contrôle": 30, "nettoyage": 45, "carie": 60, "obturation": 60, "couronne": 90,
            "bridge": 90, "blanchiment": 60, "canal": 90, "extraction": 45
        }
        for key, value in durations.items():
            if key in service_type.lower():
                return value
        return 60
