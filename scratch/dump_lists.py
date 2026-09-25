"""Check lvlOverrides + how '1.' list items are formatted in the original docx."""
import zipfile
import xml.etree.ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOCX = r"C:\Users\feede\Desktop\Rapporto.docx"
z = zipfile.ZipFile(DOCX)
r = ET.fromstring(z.read("word/numbering.xml"))

print("== lvlOverrides ==")
for num in r.findall(f"{W}num"):
    nid_el = num.find(f"{W}numId")
    nid = nid_el.get(f"{W}val") if nid_el is not None else num.get(f"{W}numId")
    for ov in num.findall(f"{W}lvlOverride"):
        ilvl = ov.get(f"{W}ilvl")
        lo = ov.find(f"{W}lvl")
        if lo is None:
            print(f"numId {nid} ilvl {ilvl}: (empty override)")
            continue
        sf = lo.find(f"{W}startOverride")
        fmt = lo.find(f"{W}numFmt")
        lt = lo.find(f"{W}lvlText")
        print(f"numId {nid} ilvl {ilvl}: start={sf.get(f'{W}val') if sf is not None else None} "
              f"fmt={fmt.get(f'{W}val') if fmt is not None else None} "
              f"text={lt.get(f'{W}val') if lt is not None else None}")

print("\n== paragraphs mentioning list anchors ==")
body = ET.fromstring(z.read("word/document.xml")).find(f"{W}body")
for p in body.iter(f"{W}p"):
    t = "".join(x.text or "" for x in p.iter(f"{W}t"))
    for anchor in ("Confronto numerico", "Ordine delle classi", "Valore effettivamente", "Guardrail a runtime"):
        if anchor in t:
            ppr = p.find(f"{W}pPr")
            npr = ppr.find(f"{W}numPr") if ppr is not None else None
            nid = npr.find(f"{W}numId").get(f"{W}val") if npr is not None and npr.find(f"{W}numId") is not None else None
            ilvl = npr.find(f"{W}ilvl").get(f"{W}val") if npr is not None and npr.find(f"{W}ilvl") is not None else None
            st = ppr.find(f"{W}pStyle").get(f"{W}val") if ppr is not None and ppr.find(f"{W}pStyle") is not None else None
            print(f"numId={nid} ilvl={ilvl} pStyle={st} | {t[:70]}")
            break
