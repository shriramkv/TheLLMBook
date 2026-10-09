import json
from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError

class Claim(BaseModel):
    policy_number: Optional[str] = Field(None, pattern=r"^[A-Z]{3}-\d{4,5}(-\d{2})?$")
    claim_type: Literal["MOTOR", "HEALTH", "PROPERTY", "TRAVEL"]
    incident_date: Optional[date] = None
    summary: str = Field(max_length=200)

# The JSON schema you can pass to a provider's structured-output feature

def parse_claim(raw_text):
    """Validate a model's reply; return (claim, None) or (None, error message)."""
    try:
        return Claim.model_validate_json(raw_text), None
    except ValidationError as e:
        return None, str(e.errors()[0]["loc"]) + ": " + e.errors()[0]["msg"]

