import joblib
from pathlib import Path
from pydantic import BaseModel
from fastapi import APIRouter

MODEL_PATH = Path(__file__).parent / "incident_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

LEGAL_KB = {
    "theft": {
        "reply": "If your handbag, phone, wallet, or other belongings were stolen or you were robbed, write down the time, place, and description of the person or vehicle involved. Do not confront the thief. Report the theft to the nearest police station and preserve CCTV footage, receipts, photos, or messages that may help.",
        "links": [
            "https://nalsa.gov.in/",
            "https://cybercrime.gov.in/"
        ],
        "urgency": "high"
    },
    "harassment": {
        "reply": "This may involve harassment. Save screenshots, messages, call records, dates, names, and witness details. Do not respond to threats. You can report the matter to the police or seek legal help.",
        "links": [
            "https://nalsa.gov.in/",
            "https://legislative.gov.in/"
        ],
        "urgency": "high"
    },
    "workplace_harassment": {
        "reply": "This may be workplace harassment. Preserve messages, emails, dates, names, and witness information. Report it to HR or management, and consider legal help if needed.",
        "links": [
            "https://nalsa.gov.in/"
        ],
        "urgency": "high"
    },
    "salary_dispute": {
        "reply": "For unpaid or delayed salary, keep your appointment letter, payslips, bank statements, attendance records, and employer messages. Send a written request to HR or management. If unresolved, seek legal help.",
        "links": [
            "https://nalsa.gov.in/"
        ],
        "urgency": "medium"
    },
    "tenant_dispute": {
        "reply": "For a tenant or landlord issue, keep the rent agreement, receipts, messages, photos, and written notices. Do not vacate or hand over keys without understanding your agreement. Consider legal advice.",
        "links": [
            "https://nalsa.gov.in/"
        ],
        "urgency": "medium"
    },
    "consumer_complaint": {
        "reply": "For a faulty product or poor service, keep the bill, order ID, screenshots, warranty details, photos, and seller messages. Complain to the seller in writing first. If unresolved, use consumer redressal channels.",
        "links": [
            "https://consumerhelpline.gov.in/"
        ],
        "urgency": "medium"
    },
    "cybercrime": {
        "reply": "This may be a cybercrime issue. Do not delete messages, screenshots, links, transaction IDs, or account details. Change passwords, enable two-factor authentication, and report through the official cybercrime portal.",
        "links": [
            "https://cybercrime.gov.in/"
        ],
        "urgency": "high"
    },
    "fraud": {
        "reply": "This may be fraud. Stop further payments immediately. Save receipts, transaction IDs, chats, phone numbers, links, and bank statements. Report it to your bank and the cybercrime portal.",
        "links": [
            "https://cybercrime.gov.in/"
        ],
        "urgency": "high"
    },
    "general_legal_help": {
        "reply": "I can help with general legal-awareness information. Please describe what happened, including dates, people involved, messages or documents, and the outcome you want.",
        "links": [
            "https://indiacode.nic.in/",
            "https://nalsa.gov.in/"
        ],
        "urgency": "low"
    }
}
KEYWORD_INTENTS = {
    "theft": [
        "stolen", "steal", "stole", "robbed", "robbery", "theft",
        "handbag", "wallet", "phone stolen", "bag stolen", "money stolen"
    ],
    "harassment": [
        "harassing", "harassment", "harass", "abusing", "abuse",
        "threatening", "threat", "bullying", "bully"
    ],
    "salary_dispute": [
        "salary", "wages", "payment not received", "unpaid", "not paid"
    ],
    "tenant_dispute": [
        "landlord", "tenant", "rent", "security deposit", "deposit"
    ],
    "consumer_complaint": [
        "fake product", "damaged product", "refund", "seller", "warranty"
    ],
    "cybercrime": [
        "hacked", "hack", "blackmail", "online fraud", "cybercrime"
    ],
    "fraud": [
        "cheated", "scam", "fraud", "fake job", "fake offer"
    ]
}

EMERGENCY_WORDS = [
    "rape",
    "suicide",
    "kill",
    "murder",
    "attack",
    "violence",
    "beating",
    "immediate danger",
    "help me now"
]


class ChatRequest(BaseModel):
    message: str


chat_router = APIRouter()


@chat_router.post("/chat")
def chat(request: ChatRequest):
    text = request.message.lower().strip()

    if any(word in text for word in EMERGENCY_WORDS):
        return {
            "intent": "emergency",
            "confidence": 1.0,
            "reply": "If you are in immediate danger, call 112 in India now. Move to a safe place if possible and contact someone you trust. This app provides general legal awareness only and does not contact emergency services.",
            "links": [
                "https://nalsa.gov.in/"
            ],
            "urgency": "critical"
        }

    if model is None:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "reply": "I can help with general legal-awareness questions. Please describe your issue in more detail.",
            "links": [],
            "urgency": "low"
        }

        text = request.message.lower()
    intent = "general_legal_help"

    for possible_intent, words in KEYWORD_INTENTS.items():
        if any(word in text for word in words):
            intent = possible_intent
            break

    confidence = 1.0 if intent != "general_legal_help" else 0.0

    info = LEGAL_KB.get(intent, LEGAL_KB["general_legal_help"])

    return {
        "intent": intent,
        "confidence": round(float(confidence), 3),
        "reply": info["reply"],
        "links": info["links"],
        "urgency": info["urgency"]
    }