import datetime
from typing import List, Dict, Any, Optional
from loguru import logger
import asyncio

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from dateutil import parser
from dateutil.tz import gettz


def _format_datetime_for_google(dt: datetime.datetime):
    """Format datetime object for Google Calendar API."""
    return dt.strftime('%Y-%m-%dT%H:%M:%S')


class GoogleCalendarService:
    """Service to interact with Google Calendar API for appointment scheduling."""

    def __init__(self, credentials_file: str, calendar_id: str):
        """Initialize Google Calendar service."""
        self.credentials_file = credentials_file
        self.calendar_id = calendar_id
        self.service = None
        self._setup_service()
        self.timezone = 'Europe/Paris'
        self.tz = gettz(self.timezone)

    def _setup_service(self):
        try:
            credentials = Credentials.from_service_account_file(
                self.credentials_file,
                scopes=['https://www.googleapis.com/auth/calendar']
            )

            self.service = build('calendar', 'v3', credentials=credentials)
            logger.info("Google Calendar service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Calendar service: {e}")
            raise

    async def get_available_slots(self, date: datetime.date, service_duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get available appointment slots for a specific date."""

        business_hours = {
            0: (9, 17),  # Monday: 9:00 to 17:00
            1: (9, 17),  # Tuesday: 9:00 to 17:00
            2: (9, 17),  # Wednesday: 9:00 to 17:00
            3: (9, 17),  # Thursday: 9:00 to 17:00
            4: (9, 17),  # Friday: 9:00 to 17:00
            5: None,  # Saturday - closed
            6: None  # Sunday - closed
        }

        weekday = date.weekday()
        if business_hours[weekday] is None:
            logger.info(f"No available slots on {date} as the office is closed")
            return []

        start_hour, end_hour = business_hours[weekday]

        # Calculate the time range for the day with timezone
        start_datetime = datetime.datetime.combine(date, datetime.time(start_hour, 0)).replace(tzinfo=self.tz)
        end_datetime = datetime.datetime.combine(date, datetime.time(end_hour, 0)).replace(tzinfo=self.tz)

        # Format for Google Calendar API
        start_str = start_datetime.isoformat()
        end_str = end_datetime.isoformat()

        logger.debug(f"Checking availability for date {date} from {start_str} to {end_str}")

        # Get busy periods from Google Calendar
        busy_periods = await self._get_busy_periods(start_datetime, end_datetime)

        # Generate available slots
        available_slots = self._generate_available_slots(
            start_datetime,
            end_datetime,
            busy_periods,
            service_duration_minutes
        )

        logger.info(f"Final available slots for {date}: {[(slot['start'], slot['end']) for slot in available_slots]}")
        return available_slots

    async def _get_busy_periods(self, start_datetime: datetime.datetime, end_datetime: datetime.datetime) -> List[
        tuple]:
        """Get busy periods from the calendar for the specified time range."""
        try:
            # Get existing events for the day
            events_result = self.service.events().list(
                calendarId=self.calendar_id,
                timeMin=start_datetime.isoformat(),
                timeMax=end_datetime.isoformat(),
                singleEvents=True,
                orderBy='startTime',
                timeZone=self.timezone
            ).execute()

            events = events_result.get('items', [])
            logger.debug(f"Found {len(events)} existing events")

            # Create and merge busy periods
            busy_periods = []
            for event in events:
                try:
                    # Extract start and end times
                    if 'dateTime' in event['start']:
                        event_start = parser.parse(event['start']['dateTime'])
                        event_end = parser.parse(event['end']['dateTime'])
                    else:
                        # All-day events
                        event_start = datetime.datetime.combine(
                            parser.parse(event['start']['date']).date(),
                            datetime.time.min
                        ).replace(tzinfo=self.tz)
                        event_end = datetime.datetime.combine(
                            parser.parse(event['end']['date']).date(),
                            datetime.time.max
                        ).replace(tzinfo=self.tz)

                    # Add to busy periods
                    busy_periods.append((event_start, event_end))
                    logger.debug(f"Added busy period: {event_start} - {event_end}")
                except Exception as e:
                    logger.error(f"Error parsing event time: {e}")
                    continue

            # Sort and merge overlapping periods
            return self._merge_overlapping_periods(busy_periods)

        except Exception as e:
            logger.error(f"Error fetching events from Google Calendar: {e}", exc_info=True)
            return []

    def _merge_overlapping_periods(self, periods: List[tuple]) -> List[tuple]:
        """Merge overlapping time periods."""
        if not periods:
            return []

        # Sort by start time
        sorted_periods = sorted(periods, key=lambda x: x[0])

        # Merge overlapping periods
        merged = [sorted_periods[0]]
        for period in sorted_periods[1:]:
            if period[0] <= merged[-1][1]:
                # Overlapping periods, merge them
                merged[-1] = (merged[-1][0], max(merged[-1][1], period[1]))
            else:
                # Non-overlapping period, add to result
                merged.append(period)

        return merged

    def _generate_available_slots(self,
                                  start_datetime: datetime.datetime,
                                  end_datetime: datetime.datetime,
                                  busy_periods: List[tuple],
                                  duration_minutes: int) -> List[Dict[str, Any]]:
        """Generate available time slots around busy periods."""
        # Setup time increments
        slot_duration = datetime.timedelta(minutes=duration_minutes)
        slot_interval = datetime.timedelta(minutes=30)

        available_slots = []
        current_time = start_datetime

        # Generate slots at regular intervals
        while current_time + slot_duration <= end_datetime:
            slot_end = current_time + slot_duration

            # Check if this slot conflicts with any busy period
            is_available = True
            for busy_start, busy_end in busy_periods:
                # Check for overlap
                if current_time < busy_end and slot_end > busy_start:
                    is_available = False
                    break

            if is_available:
                available_slots.append({
                    'start': current_time.strftime('%H:%M'),
                    'end': slot_end.strftime('%H:%M'),
                    'start_datetime': current_time,
                    'end_datetime': slot_end
                })

            current_time += slot_interval

        return available_slots

    async def create_appointment(self,
                                 start_time: datetime.datetime,
                                 end_time: datetime.datetime,
                                 patient_name: str,
                                 service_type: str = "Rendez-vous dentaire",
                                 patient_email: Optional[str] = None,
                                 patient_phone: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create a new appointment in the calendar."""
        if start_time >= end_time:
            logger.error("Invalid appointment: start time is after or equal to end time.")
            return None

        event = {
            'summary': f"{service_type.capitalize()} - {patient_name}",
            'description': f"Patient: {patient_name}\nService: {service_type}\n" +
                           (f"Email: {patient_email}\n" if patient_email else "") +
                           (f"Téléphone: {patient_phone}" if patient_phone else ""),
            'start': {
                'dateTime': _format_datetime_for_google(start_time),
                'timeZone': self.timezone,
            },
            'end': {
                'dateTime': _format_datetime_for_google(end_time),
                'timeZone': self.timezone,
            },
        }

        def _create_event():
            return self.service.events().insert(calendarId=self.calendar_id, body=event).execute()

        try:
            created_event = await asyncio.to_thread(_create_event)
            logger.info(f"Appointment created successfully: {created_event.get('htmlLink')}")
            return {
                'id': created_event.get('id'),
                'link': created_event.get('htmlLink'),
                'start': start_time.strftime('%Y-%m-%d %H:%M'),
                'end': end_time.strftime('%Y-%m-%d %H:%M'),
                'patient': patient_name,
                'service': service_type
            }
        except Exception as e:
            logger.error(f"Failed to create appointment: {e}", exc_info=True)
            return None