# fetch_gdocs.py
import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build

load_dotenv()

# Google Service Account JSON file (download from GCP console)
GOOGLE_CREDS_FILE = os.getenv("GOOGLE_CREDS_FILE")
# Comma-separated list of Google Doc IDs to index
GOOGLE_DOC_IDS = os.getenv("GOOGLE_DOC_IDS", "").split(",")


def fetch_doc_content(doc_id, service):
    """Fetch full plain text from a Google Doc."""
    doc = service.documents().get(documentId=doc_id).execute()
    content = []

    def read_elements(elements):
        text_out = []
        for elem in elements:
            if "paragraph" in elem:
                parts = elem["paragraph"].get("elements", [])
                for part in parts:
                    if "textRun" in part:
                        txt = part["textRun"].get("content", "")
                        if txt.strip():
                            text_out.append(txt)
            elif "table" in elem:
                for row in elem["table"].get("tableRows", []):
                    for cell in row.get("tableCells", []):
                        text_out.extend(read_elements(cell.get("content", [])))
            elif "tableOfContents" in elem:
                text_out.extend(read_elements(elem["tableOfContents"].get("content", [])))
        return text_out

    body = doc.get("body", {}).get("content", [])
    content.extend(read_elements(body))

    return "".join(content).strip()


def fetch_gdocs_docs():
    """Fetch all configured Google Docs as text docs."""
    if not GOOGLE_CREDS_FILE or not os.path.exists(GOOGLE_CREDS_FILE):
        print("⚠️ Missing GOOGLE_CREDS_FILE in .env or file not found.")
        return []

    if not GOOGLE_DOC_IDS or GOOGLE_DOC_IDS == [""]:
        print("⚠️ No GOOGLE_DOC_IDS configured in .env")
        return []

    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_CREDS_FILE,
        scopes=["https://www.googleapis.com/auth/documents.readonly"],
    )
    service = build("docs", "v1", credentials=creds)

    docs = []
    for doc_id in GOOGLE_DOC_IDS:
        doc_id = doc_id.strip()
        if not doc_id:
            continue

        try:
            print(f"📄 Fetching Google Doc: {doc_id}")
            text = fetch_doc_content(doc_id, service)
            if text:
                docs.append({"id": doc_id, "text": text})
            else:
                print(f"⚠️ Google Doc {doc_id} was empty, skipping.")
        except Exception as e:
            print(f"⚠️ Failed to fetch Google Doc {doc_id}: {e}")

    return docs
