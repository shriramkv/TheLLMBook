import hashlib, yaml
from jinja2 import Environment, StrictUndefined

env = Environment(undefined=StrictUndefined)     # fail loudly if a variable is missing

def load_prompt(path):
    with open(path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    raw = (spec["system"] + spec["user"]).encode("utf-8")
    spec["fingerprint"] = hashlib.sha256(raw).hexdigest()[:12]   # detects silent edits
    return spec

def render(spec, **variables):
    return [
        {"role": "system", "content": env.from_string(spec["system"]).render(**variables)},
        {"role": "user", "content": env.from_string(spec["user"]).render(**variables)},
    ]

