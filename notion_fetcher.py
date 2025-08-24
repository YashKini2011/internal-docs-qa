import os
import requests
from dotenv import load_dotenv

load_dotenv()

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_IDS = os.getenv("NOTION_IDS", "").split(",")  # can be DBs or page IDs

BASE_URL = "https://api.notion.com/v1"
HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Notion-Version": "2022-06-28",
}


def _extract_text_from_block(block):
    """Extract plain text from a Notion block."""
    block_type = block.get("type")
    if not block_type:
        return ""

    if block_type in [
        "paragraph", "heading_1", "heading_2", "heading_3",
        "to_do", "bulleted_list_item", "numbered_list_item",
        "quote", "callout", "toggle"
    ]:
        rich_text = block.get(block_type, {}).get("rich_text", [])
        return "".join([t.get("plain_text", "") for t in rich_text]).strip()

    if block_type == "code":
        code_text = block.get("code", {}).get("rich_text", [])
        return "".join([t.get("plain_text", "") for t in code_text]).strip()

    if block_type == "bookmark":
        return block.get("bookmark", {}).get("url", "")

    if block_type == "child_page":
        return block.get("child_page", {}).get("title", "")

    return ""


def _get_block_children(block_id, texts):
    """Recursively fetch children blocks."""
    url = f"{BASE_URL}/blocks/{block_id}/children?page_size=100"
    while url:
        res = requests.get(url, headers=HEADERS)
        if res.status_code != 200:
            print(f"⚠️ Failed to fetch blocks for {block_id}: {res.text}")
            return

        data = res.json()
        for block in data.get("results", []):
            text = _extract_text_from_block(block)
            if text:
                texts.append(text)
            if block.get("has_children"):
                _get_block_children(block["id"], texts)

        next_cursor = data.get("next_cursor")
        url = f"{BASE_URL}/blocks/{block_id}/children?start_cursor={next_cursor}&page_size=100" if next_cursor else None


def fetch_page_content(page_id: str) -> str:
    """Fetch a single Notion page (including children)."""
    texts = []

    # Add page title
    page_res = requests.get(f"{BASE_URL}/pages/{page_id}", headers=HEADERS)
    if page_res.status_code == 200:
        page_data = page_res.json()
        props = page_data.get("properties", {})
        for prop in props.values():
            if prop.get("type") == "title":
                title = "".join([t.get("plain_text", "") for t in prop["title"]])
                if title:
                    texts.append(f"# {title}")
                break

    # Add blocks
    _get_block_children(page_id, texts)

    return "\n".join(texts).strip()


def fetch_database_content(database_id: str):
    """Fetch all pages from a Notion database."""
    docs = []
    cursor = None
    while True:
        url = f"{BASE_URL}/databases/{database_id}/query"
        payload = {"start_cursor": cursor} if cursor else {}
        res = requests.post(url, headers=HEADERS, json=payload)
        if res.status_code != 200:
            print(f"⚠️ Error fetching database {database_id}: {res.text}")
            return []

        data = res.json()
        for row in data.get("results", []):
            row_id = row["id"]
            print(f"   ↳ Found page: {row_id}")
            content = fetch_page_content(row_id)
            if content:
                docs.append({"id": row_id, "text": content})

        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")

    return docs


def fetch_notion_docs():
    """Main entrypoint: load all docs from provided Notion IDs (pages or DBs)."""
    docs = []
    for notion_id in NOTION_IDS:
        notion_id = notion_id.strip()
        if not notion_id:
            continue

        # Try database
        res = requests.get(f"{BASE_URL}/databases/{notion_id}", headers=HEADERS)
        if res.status_code == 200:
            print(f"✅ Database found: {notion_id}")
            docs.extend(fetch_database_content(notion_id))
            continue

        # Try page
        res = requests.get(f"{BASE_URL}/pages/{notion_id}", headers=HEADERS)
        if res.status_code == 200:
            print(f"📄 Page found: {notion_id}")
            content = fetch_page_content(notion_id)
            if content:
                docs.append({"id": notion_id, "text": content})
            continue

        print(f"⚠️ Could not load ID: {notion_id}")

    return docs