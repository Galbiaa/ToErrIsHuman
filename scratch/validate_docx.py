"""Validate Rapporto_finale.docx."""
import zipfile
import xml.etree.ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
OUT = r"c:\Users\feede\Desktop\Uni\ToErrIsHuman\Rapporto_finale.docx"

z = zipfile.ZipFile(OUT)
doc = ET.fromstring(z.read("word/document.xml"))
body = doc.find(f"{W}body")

paras = [p for p in body if p.tag == f"{W}p"]
tbls = [p for p in body if p.tag == f"{W}tbl"]
blips = [b.get(f"{W}embed") if False else b.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
         for b in doc.iter(f"{A}blip")]

styles = {}
for p in paras:
    ppr = p.find(f"{W}pPr")
    st = ppr.find(f"{W}pStyle").get(f"{W}val") if ppr is not None and ppr.find(f"{W}pStyle") is not None else None
    styles[st] = styles.get(st, 0) + 1

print("tables:", len(tbls))
print("images:", sorted(blips))
print("style counts:", dict(sorted(styles.items(), key=lambda x: -x[1])))

# lists
nid_count = {}
for p in paras:
    ppr = p.find(f"{W}pPr")
    npr = ppr.find(f"{W}numPr") if ppr is not None else None
    if npr is not None:
        nid = npr.find(f"{W}numId").get(f"{W}val")
        nid_count[nid] = nid_count.get(nid, 0) + 1
print("numPr counts:", nid_count)

# text spot checks
full = "\n".join(t.text or "" for t in doc.iter(f"{W}t"))
for probe in ("Artefatti principali", "Nota sull'inferenza", "Figura 9", "unified_D_metrics_table",
              "opzionale: --out path/to/out.json", "Confronto numerico", "Rapporto"):
    print(f"  contains {probe!r}:", probe in full)

# every cell of first table
t0 = tbls[0]
rows = t0.findall(f"{W}tr")
print("first table rows:", len(rows), "cols:", len(rows[0].findall(f"{W}tc")))
