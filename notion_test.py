from notion_fetcher import fetch_page_content

if __name__ == "__main__":
    test_page_id = "258c88175c3280f69f3cd9dc0a52d33c"  # replace if needed
    content = fetch_page_content(test_page_id)
    print(f"--- PAGE {test_page_id} ---\n")
    print(content[:1000] + "...\n" if len(content) > 1000 else content)
