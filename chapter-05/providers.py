"""Thin adapters: each returns {"text", "input_tokens", "output_tokens"}."""
import os, re, json

def call_anthropic(messages, model, temperature=None, max_tokens=400):
    import anthropic                                   # pip install anthropic
    client = anthropic.Anthropic()                     # reads ANTHROPIC_API_KEY
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    chat = [m for m in messages if m["role"] != "system"]
    # The current Anthropic Python SDK has no temperature argument for this call,
    # so it is deliberately not passed here.
    r = client.messages.create(model=model, system=system, messages=chat,
                               max_tokens=max_tokens)
    return {"text": r.content[0].text,
            "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}

def call_openai_compatible(messages, model, temperature=0, max_tokens=400):
    from openai import OpenAI                          # pip install openai
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))   # also works with local servers
    kwargs = {"temperature": temperature} if temperature is not None else {}
    r = client.chat.completions.create(model=model, messages=messages,
                                       max_tokens=max_tokens, **kwargs)
    return {"text": r.choices[0].message.content,
            "input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens}

def offline_stub(messages, model=None, temperature=0, max_tokens=400):
    """A rule-based stand-in for testing the workbench without an API key.
    It ignores the prompt wording, so its scores say nothing about prompt quality."""
    email = messages[-1]["content"]
    pol = re.search(r"\b[A-Z]{3}-\d{4,5}(?:-\d{2})?\b", email)
    kind = next((k for k, words in {"MOTOR": ["car", "scooter", "bike"],
                                    "HEALTH": ["hospital", "admitted"],
                                    "TRAVEL": ["flight", "baggage"],
                                    "PROPERTY": ["ceiling", "home", "burglary"]}.items()
                 if any(w in email.lower() for w in words)), None)
    d = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", email)
    out = {"policy_number": pol.group(0) if pol else None, "claim_type": kind,
           "incident_date": f"{d.group(3)}-{d.group(2)}-{d.group(1)}" if d else None,
           "summary": email[:80]}
    text = json.dumps(out)
    return {"text": text, "input_tokens": sum(len(m["content"]) for m in messages) // 4,
            "output_tokens": len(text) // 4}
