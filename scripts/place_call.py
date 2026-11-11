"""Place an outbound call via Twilio that is driven by the agent.

  pip install twilio
  export TWILIO_ACCOUNT_SID=... TWILIO_AUTH_TOKEN=... TWILIO_FROM=+1...
  export CSVA_BRAIN=gemini GEMINI_API_KEY=...
  python -m csva serve &          # then expose it:  ngrok http 8000
  python scripts/place_call.py --to +91XXXXXXXXXX --public-url https://<id>.ngrok.app --case ivr_paid

Call a number YOU control (e.g. a Twilio Studio flow that plays an IVR). Do not dial real payers with real PHI.
"""

import argparse
import os

from twilio.rest import Client

p = argparse.ArgumentParser()
p.add_argument("--to", required=True)
p.add_argument("--public-url", required=True)
p.add_argument("--case", default="ivr_denied_medical_necessity")
a = p.parse_args()

client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
call = client.calls.create(
    to=a.to,
    from_=os.environ["TWILIO_FROM"],
    url=f"{a.public_url}/twiml/start?case={a.case}",
    status_callback=f"{a.public_url}/twiml/status",
    status_callback_event=["completed"],
)
print("call sid:", call.sid)
