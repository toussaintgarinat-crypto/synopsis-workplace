def to_excalidraw(d):
    elements, y = [], 0
    for i, c in enumerate(d.get("chapitres", [])):
        label = f'{c.get("timestamp", "")} {c.get("subject", "")}'
        elements.append({"id": f"r{i}", "type": "rectangle", "x": 0, "y": y, "width": 420, "height": 60})
        elements.append({"id": f"t{i}", "type": "text", "x": 12, "y": y + 18, "text": label, "fontSize": 20})
        y += 90
    return {"type": "excalidraw", "version": 2, "source": "synopsis", "elements": elements}
