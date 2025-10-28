from flask import Flask, request, jsonify, render_template, session, send_file
from flask_cors import CORS
from datetime import timedelta
from fuzzywuzzy import fuzz, process
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pypdf import PdfReader 
from dotenv import load_dotenv
from openai import OpenAI, APIError
import os, io, random, smtplib, sys
import openai

#
# 1. KONFIGURATION & SETUP
# 
load_dotenv()

# Umgebungsvariablen sicher laden
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#openai.api_key = "PgsFg0ScaIHbrIYEVuQMfGu2kI9J6oKH9oA"
EMAIL_ABSENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORT = os.getenv("EMAIL_PASSWORT")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", os.urandom(24)) # Sicherer Fallback

# Zentrale Prüfung der kritischen Konfiguration
if not OPENAI_API_KEY:
    print("FEHLER: OPENAI_API_KEY fehlt in der .env-Datei. GPT-Funktionen sind deaktiviert.")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

if not EMAIL_PASSWORT or not EMAIL_ABSENDER:
    print("WARNUNG: E-Mail-Konfiguration unvollständig. Der E-Mail-Versand ist deaktiviert.")

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY
app.permanent_session_lifetime = timedelta(days=1)
CORS(app)

# 
# 2. RAG-DATEN (PDF-Extraktion mit Fehlerbehandlung)
# 
def extract_pdf_text(path):
    """Extrahiert Text aus einer PDF-Datei."""
    reader = PdfReader(path)
    # Verwende 'or ""' falls eine Seite keinen Text enthält
    return "\n".join(page.extract_text() or "" for page in reader.pages)

resume_text = ""
bewerbung_text = ""
try:
    #  Dateien laden
    resume_text = extract_pdf_text("lebenslauf_aras_2.pdf")
    bewerbung_text = extract_pdf_text("AS_Aras.pdf")
    print("PDFs erfolgreich geladen.")
except FileNotFoundError as e:
    print(f"FEHLER: Mindestens eine PDF-Datei (z.B. {e.filename}) nicht gefunden. Der Chatbot wird ohne spezifischen Lebenslauf-Kontext arbeiten.")
except Exception as e:
    print(f"FEHLER beim Lesen der PDFs: {e}. Kontext ist möglicherweise unvollständig.")
    
# Der gesamte Text, der an GPT gesendet wird.
RESUME_TEXT = (resume_text + "\n\n" + bewerbung_text).strip()

# 
# 3. STATISCHE ANTWORTEN (Regelwerk)
# 
antworten = {
    # GRUSSE
    ("hallo", "hi", "hey", "servus"): ["Hallo! Ich bin ARA, Aras’ persönlicher Assistent. Wie kann ich dir helfen?", "Guten Tag! Was möchtest du über Aras wissen?"],
    ("wer bist du", "wie heißt du"):["Ich bin ARA, der persönliche KI-Assistent von Aras Ismail Adaib, entwickelt für Bewerbungsfragen."],
    "wer ist aras": "Aras Ismail Adaib ist ein angehender Fachinformatiker für Anwendungsentwicklung in Dortmund.",
    "was macht aras": "Er macht eine Umschulung zum Fachinformatiker für Anwendungsentwicklung bei Cimdata Dortmund.",
    "danke": ["Gern geschehen!","Kein Problem!"],
    # PERSÖNLICHES
    "was sind seine hobbys": "Fotografie, Keyboardspielen und Gaming.",
    "was sind seine stärken": "Teamgeist, Lernbereitschaft, Kreativität und analytisches Denken.",
    # VERABSCHIEDUNG
    ("tschüss", "bye", "auf wiedersehen"): "Tschüss! Ich bleibe hier, falls du noch Fragen zu Aras hast."
}

# 
# 4. HILFSFUNKTIONEN (Logik)
# 
def fuzzy_match(frage, threshold=90): # Threshold auf 90 erhöht für präzisere Regelwerk-Treffer
    """Sucht nach der besten Übereinstimmung im Antworten-Wörterbuch."""
    frage = frage.lower().strip()
    alle, mapping = [], {}
    
    # 1. Vorbereitung der Schlüssel für das Matching
    for key, val in antworten.items():
        if isinstance(key, tuple):
            for sub in key:
                alle.append(sub); mapping[sub] = key
        else:
            alle.append(key); mapping[key] = key

    # 2. Effizientes Matching mit token_set_ratio (ignoriert Wortreihenfolge)
    res = process.extractOne(frage, alle, scorer=fuzz.token_set_ratio, score_cutoff=threshold)
    
    if not res: return None
    
    # 3. Antwortauswahl
    antwort = antworten[mapping[res[0]]]
    return random.choice(antwort) if isinstance(antwort, list) else antwort

def gpt_answer(user_input):
    """Generiert eine Antwort mithilfe der OpenAI GPT-API und dem Lebenslauf-Kontext."""
    if not client:
        return "Der KI-Service ist aufgrund fehlender Konfiguration (API-Key) deaktiviert."
        
    prompt = f"""
Du bist **ARA**, der persönliche KI-Assistent von Aras. Deine Hauptaufgabe ist es, Fragen, die im statischen Regelwerk nicht beantwortet werden konnten, zu beantworten.

Nutze den folgenden Lebenslauf- und Bewerbungsinhalt (RAG-Context), um Fragen über Aras **faktenbasiert und korrekt** zu beantworten:
--- RAG CONTEXT START ---
{RESUME_TEXT if RESUME_TEXT else 'Keine Dokumente geladen. Antworte nur mit allgemeinem Wissen über Aras.'}
--- RAG CONTEXT END ---

Regeln:
1. Antworte immer **auf Deutsch**.
2. **Wenn die Frage im Zusammenhang mit Aras' Lebenslauf steht, nutze ausschließlich die obigen Informationen.** 3. Wenn der Kontext nicht in den Dokumenten zu finden ist (z.B. allgemeine Technikfragen), antworte als sachlicher, hilfreicher KI-Assistent.
4. Sei präzise, freundlich, professionell und halte dich kurz und knap.
"""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": user_input}],
            max_tokens=200,
            temperature=0.5 # Auf 0.5 reduziert für mehr Präzision und Faktenfokus
        )
        return r.choices[0].message.content.strip()
    except APIError as e:
        return f"OpenAI-API-Fehler (Rate Limit/Auth): {e}"
    except Exception as e:
        return f"Ein unerwarteter Fehler ist aufgetreten: {e}"

def verlauf_als_text():
    """Formatiert den Chatverlauf als Text."""
    hist = session.get("history", [])
    return "\n".join(f"Frage: {h['frage']}\nAntwort: {h['antwort']}\n" for h in hist)

# ---------------------------------------------------------------------------
# 5. ROUTEN
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", bot_name="ARA")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"reply": "Bitte gib eine Nachricht ein."}), 400

    if "history" not in session:
        session["history"] = []

    # 1. Rule-Based (Fuzzy Match)
    antwort = fuzzy_match(user_input)
    
    # 2. Fallback to GPT
    if not antwort:
        antwort = gpt_answer(user_input)

    # 3. Speichern und Antworten
    session["history"].append({"frage": user_input, "antwort": antwort})
    session.modified = True

    return jsonify({
        "reply": antwort,
        # Prüfe, ob die Frage in der gesamten Historie außer dem letzten Eintrag gestellt wurde
        "asked_before": any(h["frage"] == user_input for h in session["history"][:-1])
    })

@app.route("/history")
def history():
    return jsonify(session.get("history", []))

@app.route("/clear-history", methods=["POST"])
def clear_history():
    session["history"] = []
    session.modified = True
    return jsonify({"message": "Verlauf gelöscht."})

@app.route("/download-history")
def download_history():
    data = verlauf_als_text()
    return send_file(io.BytesIO(data.encode("utf-8")),
                     mimetype="text/plain",
                     as_attachment=True,
                     download_name="chatverlauf.txt")

@app.route("/send-email", methods=["POST"])
def send_email():
    data = request.json
    empfaenger = data.get("email")
    if not empfaenger:
        return jsonify({"success": False, "message": "Keine E-Mail-Adresse angegeben."}), 400
        
    text_body = verlauf_als_text()
    if not text_body:
        return jsonify({"success": False, "message": "Kein Verlauf vorhanden."}), 400
        
    # Überprüfung der E-Mail-Konfiguration
    if not EMAIL_PASSWORT or not EMAIL_ABSENDER:
         return jsonify({"success": False, "message": "E-Mail-Dienst nicht konfiguriert (Passwort/Absender fehlt)."}), 500

    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = EMAIL_ABSENDER, empfaenger, "Dein Chatverlauf mit ARA"
        msg.attach(MIMEText(text_body, "plain", "utf-8"))

        # Mail-Server-Details auslagern oder hier als Konstanten verwenden
        smtp_server = "smtp.web.de"
        smtp_port = 587

        with smtplib.SMTP(smtp_server, smtp_port) as s:
            s.starttls()
            s.login(EMAIL_ABSENDER, EMAIL_PASSWORT)
            s.send_message(msg)

        return jsonify({"success": True, "message": "E-Mail erfolgreich gesendet."})
    except Exception as e:
        # Fängt Fehler wie Authentifizierung, Server-Verbindung usw. ab
        return jsonify({"success": False, "message": f"E-Mail-Fehler: {e}. Bitte prüfen Sie die App-Passwort-Einstellungen."}), 500

# 
# 6. START
#
if __name__ == "__main__":
    # Stellt sicher, dass die Anwendung läuft, auch wenn PDFs fehlen.
    app.run(debug=True)