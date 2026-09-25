#!/usr/bin/env python
"""Inspect the structure of Rapporto.docx (original template)."""
import zipfile
import sys
import os
import re
import xml.etree.ElementTree as ET

DOCX = r"C:\Users\feede\Desktop\Rapporto.docx"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

print(f"File exists: {os.path.exists(DOCX)}")
print(f"File size: {os.path.getsize(DOCX)} bytes")
print()

with zipfile.ZipFile(DOCX) as z:
    for info in z.infolist():
        print(f"{info.file_size:>10}  {info.filename}")
print()

# Inspect document.xml
with open(r"C:\Users\feede\Desktop\Rapporto\word\document.xml", encoding="utf-8") as f:
    content = f.read()

# Find all drawing elements
drawings = re.findall(r"<w:drawing>.*?</w:drawing>", content, re.S)
print(f"Total drawing elements in document.xml: {len(drawings)}")
for d in drawings:
    rids = re.findall(r'r:embed="(rId\d+)"', d)
    print(f"  r:embed = {rids}")

# Show figure elements (style 907 - Captioned Figure)
print()
tree = ET.parse(r"C:\Users\feede\Desktop\Rapporto\word\document.xml")
root = tree.getroot()
body = root[0]
for i, child in enumerate(body):
    if child.tag != f"{W}p":
        continue
    pPr = child.find(f"{W}pPr")
    if pPr is not None:
        pStyle = pPr.find(f"{W}pStyle")
        if pStyle is not None and pStyle.get(f"{W}val") in ("907", "906"):
            txts = [t.text for t in child.iter(f"{W}t") if t.text]
            txt = " ".join(txts)[:100] if txts else ""
            xml_str = ET.tostring(child, encoding="unicode")[:300]
            print(f"body[{i}] style={pStyle.get(f'{W}val')} -> {xml_str}")

# Count tables
tbl_count = sum(1 for c in body if c.tag == f"{W}tbl")
print(f"\nTotal tbl elements in body: {tbl_count}")