"""Lab 10.1, Part C (Section 10.6): read a photographed job card into a validated record.

Every field is optional, and the model must list any field it cannot read rather than guess
(the same idea as the claims extractor in Section 5.4). Cards with unreadable fields, or
replies that fail validation, go to a person.

    pip install openai pydantic
    export OPENAI_API_KEY=... VISION_MODEL=gpt-4o-mini
    python lab_10_1_part_c_job_card.py samples/job_card.jpg
    python lab_10_1_part_c_job_card.py --offline       # a canned reply, to test validation and routing

The sample card has two traps: the 'Time in / out' line is smudged, and the handwritten
'B52' can be read as 'BS2'. Check whether your model flags them or guesses.
"""
import argparse, base64, json, os
from typing import Optional
from pydantic import BaseModel, Field, ValidationError

class JobCard(BaseModel):
    card_number: Optional[str] = None
    date: Optional[str] = Field(None, description="ISO format YYYY-MM-DD; the card is written DD/MM/YYYY")
    site: Optional[str] = None
    equipment: Optional[str] = None
    fault_reported: Optional[str] = None
    work_done: Optional[str] = None
    parts_used: list[str] = Field(default_factory=list)
    time_in: Optional[str] = Field(None, description="24-hour HH:MM")
    time_out: Optional[str] = Field(None, description="24-hour HH:MM")
    technician: Optional[str] = None
    customer_signed: Optional[bool] = None
    unreadable_fields: list[str] = Field(default_factory=list,
        description="names of fields that are present on the card but cannot be read with confidence")

PROMPT = f"""Read this photographed field service job card and return one JSON object that
matches this JSON schema:
{json.dumps(JobCard.model_json_schema(), indent=1)}
Copy values exactly as written. If a field is blank, use null. If a field is written but you
cannot read it with confidence, use null AND add the field name to unreadable_fields.
Never guess. Return only the JSON object."""

CANNED = ('{"card_number": "04417", "date": "2026-10-11", "site": "Prestige Tech Park, Block B", '
          '"equipment": "AHU-3 (roof)", "fault_reported": "Belt squeal at start-up", '
          '"work_done": "Replaced both belts, set tension", "parts_used": ["2 x B52 V-belt (AB-B52-MS)"], '
          '"time_in": null, "time_out": null, "technician": "Mahesh R.", "customer_signed": true, '
          '"unreadable_fields": ["time_in", "time_out"]}')

def read_card(path, offline=False):
    if offline:
        print("[stand-in] offline: using a canned model reply\n")
        return CANNED
    from openai import OpenAI
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    reply = OpenAI().chat.completions.create(
        model=os.environ["VISION_MODEL"], max_tokens=600, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}])
    return reply.choices[0].message.content

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", default="samples/job_card.jpg")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    raw = read_card(args.image, args.offline)
    try:
        card = JobCard.model_validate_json(raw)
    except ValidationError as e:
        print("Reply failed validation: route to a person.\n", e)
        raise SystemExit(1)
    print(card.model_dump_json(indent=2))
    if card.unreadable_fields:
        print(f"\nROUTE TO OFFICE: unreadable {', '.join(card.unreadable_fields)}")
    else:
        print("\nAll fields read: record can be filed automatically (spot-check a sample).")
