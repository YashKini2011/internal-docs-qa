from notion_fetcher import load_notion_docs

if __name__ == "__main__":
    docs = load_notion_docs()
    print(f"\n✅ Loaded {len(docs)} documents\n")

    for i, d in enumerate(docs, 1):
        print(f"--- Document {i} ---")
        print(d[:500] + "...\n" if len(d) > 500 else d)
