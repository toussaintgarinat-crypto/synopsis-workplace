def summarize_many(urls, summarize):
    out = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        try:
            out.append({"url": url, "ok": True, "resultat": summarize(url)})
        except Exception as e:
            out.append({"url": url, "ok": False, "erreur": str(e)})
    return out
