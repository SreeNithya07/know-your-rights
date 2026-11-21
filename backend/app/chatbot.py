import joblib
from pathlib import Path
from pydantic import BaseModel
from fastapi import APIRouter
from app.rag.retriever import search_legal_documents


# Load the trained legal provision ML model
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LEGAL_MODEL_PATH = (
    PROJECT_ROOT / "ml" / "artifacts" / "legal_provision_model.joblib"
)

try:
    legal_bundle = joblib.load(LEGAL_MODEL_PATH)
    legal_model = legal_bundle["model"]
    legal_encoder = legal_bundle["encoder"]
    LEGAL_THRESHOLD = legal_bundle["threshold"]

    print("Legal provision ML model loaded successfully!")

except Exception as error:
    print("Legal ML model loading failed:", error)
    legal_model = None
    legal_encoder = None
    LEGAL_THRESHOLD = 0.45

MODEL_PATH = Path(__file__).parent / "incident_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

LEGAL_KB = {
    "public_smoking": {
        "reply": "Smoking in public places and public transport may be restricted under applicable rules. Note the date, time, vehicle or location, and take a photo or video only if it is safe. You may report repeated violations to the transport authority, local civic body, or police. Avoid confrontation if the person becomes aggressive.",
        "links": [
            "https://www.india.gov.in/",
            "https://www.indiacode.nic.in/"
        ],
        "urgency": "low"
    },
    "civic_issue": {
        "reply": "For civic problems such as garbage, damaged roads, street lights, water supply, drainage, or stray animals, record the exact location, date, time, photos, and videos. Report it to your local municipal body, panchayat, or relevant department. Keep the complaint number and follow up if the issue is not resolved.",
        "links": [
            "https://pgportal.gov.in/",
            "https://www.india.gov.in/"
        ],
        "urgency": "medium"
    },
    "government_misuse": {
        "reply": "If a police officer, government employee, or public representative is taking money, accepting a bribe, or misusing their power, do not pay or confront them directly. Write down the officer or employee’s name, designation, badge or ID number, office, date, time, location, amount demanded, and any vehicle or receipt details. Preserve messages, call recordings, photos, videos, and witness information. You may file a public grievance through CPGRAMS or approach the concerned department, vigilance authority, or appropriate anti-corruption authority.",
        "links": [
            "https://pgportal.gov.in/",
            "https://indiacode.gov.in/"
        ],
        "urgency": "high"
    },
    "kidnapping": {
        "reply": "If someone has been kidnapped, abducted, or is being held against their will, contact the police immediately. If there is immediate danger, call 112. Preserve the last known location, phone number, photo, clothing description, vehicle details, and any messages or calls. Do not try to confront the kidnapper yourself.",
        "links": [
            "https://www.indiacode.nic.in/"
        ],
        "urgency": "critical"
    },
    "public_nuisance": {
        "reply": "For public nuisance, record the date, time, location, type of disturbance, and any photos, videos, or witness details. You may report repeated disturbances to the local police, municipal authority, or relevant local body. Avoid direct confrontation if there is a risk of violence.",
        "links": [
            "https://www.indiacode.nic.in/"
        ],
        "urgency": "medium"
    },
    "child_labor": {
        "reply": "Child labour is a serious issue. Note the child’s approximate age, work location, working hours, employer or shop name, address, and any photos or videos only if it is safe to take them. You may report it to the local police, labour department, child protection authorities, or Child Helpline 1098. Do not confront the employer or put the child at risk.",
        "links": [
            "https://www.indiacode.nic.in/",
            "https://www.wcd.gov.in/child/child-helpline"
        ],
        "urgency": "high"
    },
    "domestic_violence": {
        "reply": "If you are facing domestic violence, prioritize your safety. Move to a safe place if possible, keep evidence such as messages, medical records, photos, and witness details, and contact someone you trust. You may seek help from the police, Protection Officer, or legal-aid services. If there is immediate danger, call 112.",
        "links": [
            "https://nalsa.gov.in/",
            "https://www.indiacode.nic.in/"
        ],
        "urgency": "critical"
    },
    "dowry_harassment": {
        "reply": "If you are facing dowry demands or harassment, preserve messages, call records, bank transactions, gifts, photographs, and witness details. Do not agree to demands under pressure. You may seek help from the police, a lawyer, women’s helpline, or legal-aid services.",
        "links": [
            "https://nalsa.gov.in/",
            "https://www.indiacode.nic.in/"
        ],
        "urgency": "high"
    },
    "discrimination": {
        "reply": "Discrimination based on caste, religion, gender, disability, or other protected characteristics may require a complaint to the appropriate authority. Keep messages, names, dates, witnesses, and any evidence of unequal treatment. Legal-aid services may be able to help you understand available options.",
        "links": [
            "https://nalsa.gov.in/",
            "https://www.indiacode.nic.in/"
        ],
        "urgency": "medium"
    },
    "constitutional_rights": {
        "reply": "The Constitution of India protects fundamental rights, including equality, freedom of speech and expression, protection of life and personal liberty, and protection against exploitation. The exact rights and remedies depend on your situation. You can read the official Constitution through India Code.",
        "links": [
            "https://www.indiacode.nic.in/handle/123456789/1362",
            "https://www.indiacode.nic.in/"
        ],
        "urgency": "low"
    },
    "rti": {
        "reply": "Under the Right to Information process, you can ask a public authority for information held by it. Write a clear request, mention the information you need, and submit it to the concerned Public Information Officer. Keep a copy of your application and any response.",
        "links": [
            "https://rtionline.gov.in/",
            "https://www.indiacode.nic.in/"
        ],
        "urgency": "low"
    },
    "defamation": {
        "reply": "If someone is spreading false statements or rumours about you, save screenshots, links, dates, names, and witness details. Do not respond aggressively. A lawyer can help you understand whether the statements may be defamatory and what remedies may be available.",
        "links": [
            "https://www.indiacode.nic.in/",
            "https://nalsa.gov.in/"
        ],
        "urgency": "medium"
    },
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
        "forced_religious_conversion": {
        "reply": (
            "Forcing someone to change their religion through threats, "
            "coercion, or pressure may raise serious legal concerns. "
            "Article 25 of the Constitution protects freedom of conscience "
            "and the freedom to profess, practise and propagate religion, "
            "subject to constitutional restrictions. Depending on the facts "
            "and the state where this occurred, specific laws concerning "
            "unlawful religious conversion may also apply. Preserve threatening "
            "messages and other evidence if it is safe. You may approach the "
            "police or seek free legal assistance through NALSA."
        ),
        "links": [
            "https://nalsa.gov.in/",
            "https://www.indiacode.nic.in/"
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
    "public_smoking": [
        "smoking in public", "smoking in bus", "smoking in train",
        "smoking in public transport", "smoking in public place",
        "someone smoking", "cigarette in public"
    ],
    "civic_issue": [
        "garbage", "street light", "street lights", "road damage",
        "damaged road", "water supply", "no water", "electricity",
        "no electricity", "stray dogs", "drainage", "sewage",
        "pothole", "public toilet", "municipality"
    ],
    "government_misuse": [
        "government employee", "public servant", "misusing power",
        "misuse of power", "bribe", "bribery", "corruption",
        "police misuse", "officer misuse", "police taking money",
        "taking money", "accepting money", "accepting bribe",
        "asking for money", "demanding money", "culprits",
        "politician misusing", "politician in power"
    ],
    "kidnapping": [
        "kidnap", "kidnapped", "kidnapping", "abduct", "abducted",
        "missing", "missing person", "missing child", "missing kid",
        "my kid is missing", "my child is missing", "held against my will",
        "captivity"
    ],
    "public_nuisance": [
        "public nuisance", "loud music", "noise complaint",
        "public disturbance", "blocking road", "public place"
    ],
    "child_labor": [
        "child labour", "child labor", "child working", "child exploitation",
        "forced to beg", "child beggar", "child working in factory",
        "child working in restaurant", "child working in tea stall",
        "child working in shop", "children working"
    ],
    "domestic_violence": [
        "domestic violence", "husband abusing", "family abuse",
        "wife beating", "domestic abuse", "violence at home"
    ],
    "dowry_harassment": [
        "dowry", "dowry harassment", "dowry demand", "misbehave"
    ],
    "discrimination": [
        "discrimination", "caste discrimination", "religion discrimination",
        "gender discrimination", "untouchability"
    ],
    "constitutional_rights": [
        "fundamental rights", "constitutional rights", "indian constitution",
        "my rights under constitution"
    ],
    "rti": [
        "rti", "right to information", "information officer",
        "government information"
    ],
    "defamation": [
        "defamation", "false rumors", "fake news about me",
        "spreading rumours", "spreading rumors"
    ],
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


def predict_legal_provisions(message):
    if legal_model is None:
        return []

    try:
        probabilities = legal_model.predict_proba([message])[0]

        predictions = []

        for index, probability in enumerate(probabilities):
            if probability >= LEGAL_THRESHOLD:
                predictions.append({
                    "provision": str(legal_encoder.classes_[index]),
                    "confidence": round(float(probability), 3)
                })

        predictions.sort(
            key=lambda item: item["confidence"],
            reverse=True
        )

        return predictions[:5]

    except Exception as error:
        print("Legal prediction error:", error)
        return []


chat_router = APIRouter()


@chat_router.post("/chat")
def chat(request: ChatRequest):
    text = request.message.lower().strip()

    # 1. Emergency detection
    if any(word in text for word in EMERGENCY_WORDS):
        return {
            "intent": "emergency",
            "confidence": 1.0,
            "reply": (
                "If you are in immediate danger, call 112 in India now. "
                "Move to a safe place if possible and contact someone "
                "you trust. This app provides general legal awareness "
                "only and does not contact emergency services."
            ),
            "links": ["https://nalsa.gov.in/"],
            "urgency": "critical",
            "sources": []
        }

    # 2. Identify legal issues with priority-based detection

    intent = "general_legal_help"

    # Forced religious conversion
    conversion_words = [
        "force me to convert",
        "forcing me to convert",
        "forced to convert",
        "force to convert",
        "convert my religion",
        "change my religion",
        "forced religious conversion",
        "threatening me to convert",
        "forcing conversion",
        "religious conversion by force"
    ]

    # Child labour in hazardous places
    child_labour_words = [
        "child working in coal mine",
        "child working in coal mines",
        "children working in mines",
        "child labour in mines",
        "child labor in mines",
        "child working in mine",
        "kid working in coal mine",
        "child working in factory",
        "child working in hazardous",
        "child labour",
        "child labor",
        "child exploitation",
        "children working"
    ]

    # Domestic violence and spousal abuse
    domestic_words = [
        "husband harassing",
        "husband is harassing",
        "husband abusing",
        "husband is abusing",
        "husband beats",
        "husband beating",
        "husband threatens",
        "husband threatening",
        "wife being abused",
        "wife being beaten",
        "domestic violence",
        "domestic abuse",
        "family abuse",
        "violence at home",
        "in laws harassing",
        "in-laws harassing"
    ]

    if any(word in text for word in conversion_words):
        intent = "forced_religious_conversion"

    elif any(word in text for word in child_labour_words):
        intent = "child_labor"

    elif any(word in text for word in domestic_words):
        intent = "domestic_violence"

    else:
        for possible_intent, words in KEYWORD_INTENTS.items():
            if any(word in text for word in words):
                intent = possible_intent
                break

    confidence = 1.0 if intent != "general_legal_help" else 0.0

    # 3. Get the existing predefined legal response
    info = LEGAL_KB.get(
        intent,
        LEGAL_KB["general_legal_help"]
    )

    reply = info["reply"]
    links = list(info["links"])
    sources = []

    # 4. Retrieve relevant constitutional information using RAG
    if intent == "constitutional_rights":
        try:
            results = search_legal_documents(
                request.message,
                number_of_results=3
            )

            if results:
                reply += "\n\nRelevant information from the Constitution:\n"

                for index, result in enumerate(results, start=1):
                    reply += (
                        f"\n{index}. {result['text'][:900]}\n"
                    )

                    sources.append({
                        "document": result["document"],
                        "page": result["page"],
                        "category": result["category"]
                    })

                links.append(
                    "https://www.legislative.gov.in/documents"
                )

        except Exception as error:
            print("RAG retrieval error:", error)

    # 5. Predict possible legal provisions using trained ML model
    provisions = predict_legal_provisions(request.message)

    # 6. Return the response to React frontend
    return {
        "intent": intent,
        "confidence": round(float(confidence), 3),
        "reply": reply,
        "links": list(dict.fromkeys(links)),
        "urgency": info["urgency"],
        "sources": sources,
        "provisions": provisions
    }