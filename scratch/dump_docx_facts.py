"""Dump numbering/styles/table/figure facts from the original Rapporto.docx."""
import re
import zipfile
from collections import Counter

SRC = r"C:\Users\feede\Desktop\Rapporto.docx"

z = zipfile.ZipFile(SRC)
doc = z.read("word/document.xml").decode("utf-8")
num = z.read("word/numbering.xml").decode("utf-8")
sty = z.read("word/styles.xml").decode("utf-8")

# --- numbering: map numId -> abstractNumId, and abstract formats
for m in re.finditer(r'<w:num w:numId="(\d+)"[^>]*>\s*<w:abstractNumId w:val="(\d+)"/>', num):
    print(f"numId {m.group(1)} -> abstractNum {m.group(2)}")

print("\n--- abstractNum formats ---")
for m in re.finditer(r'<w:abstractNum w:abstractNumId="(\d+)"[^>]*>(.*?)</w:abstractNum>', num, re.S):
    aid, body = m.group(1), m.group(2)
    lvls = re.findall(
        r'<w:lvl w:ilvl="(\d)"[^>]*>.*?<w:numFmt w:val="([^"]+)"/>.*?<w:lvlText w:val="([^"]*)"/>',
        body, re.S)
    summary = ", ".join(f"ilvl{i}:{fmt}'{txt}'" for i, fmt, txt in lvls[:4])
    print(f"abstractNum {aid}: {summary}")

# --- style ids used by paragraphs in the original document
print("\n--- pStyle usage in original document.xml ---")
c = Counter(re.findall(r'<w:pStyle w:val="([^"]+)"/>', doc))
print(dict(c))

# --- numPr usage (numId counts)
print("\n--- numId usage ---")
c2 = Counter(re.findall(r'<w:numId w:val="(\d+)"/>', doc))
print(dict(c2))

# --- sample of first table's tblPr + first row cell props
tm = re.search(r"<w:tbl>(.*?)</w:tbl>", doc, re.S)
if tm:
    tbl = tm.group(1)
    tp = re.search(r"<w:tblPr>.*?</w:tblPr>", tbl, re.S)
    print("\n--- first table tblPr ---")
    print(tp.group(0) if tp else "none")
    grid = re.search(r"<w:tblGrid>.*?</w:tblGrid>", tbl, re.S)
    print("\n--- first table tblGrid ---")
    print(grid.group(0) if grid else "none")
    first_tr = re.search(r"<w:tr\b.*?</w:tr>", tbl, re.S)
    print("\n--- first row (truncated 2200 chars) ---")
    print(first_tr.group(0)[:2200] if first_tr else "none")

# --- figure paragraph context: pPr around a drawing
dm = re.search(r'<w:p><w:pPr>(?:(?!</w:p>).)*?<w:drawing>', doc, re.S)
if dm:
    frag = doc[dm.start():dm.start() + 900]
    print("\n--- figure paragraph head ---")
    print(frag)

# --- heading paragraph sample
for sid in ("878", "879", "880"):
    hm = re.search(rf'<w:p><w:pPr><w:pStyle w:val="{sid}"/>(.*?)(?=<w:r>|</w:p>)', doc, re.S)
    if hm:
        print(f"\n--- style {sid} pPr sample ---")
        print(hm.group(1)[:400])

# --- bullet paragraph sample (numId 2 if present)
bm = re.search(r'<w:p><w:pPr><w:pStyle w:val="868"/>(?:(?!</w:pPr>).)*?<w:numPr>.*?</w:pPr>', doc, re.S)
if bm:
    print("\n--- compact numbered/bullet paragraph sample ---")
    print(doc[bm.start():bm.start() + 900])

# --- styles.xml names for ids we plan to use
print("\n--- style names ---")
for sid in ("866", "867", "868", "878", "879", "880", "900", "905", "907", "914"):
    sm = re.search(rf'<w:style w:type="[^"]*"[^>]*w:styleId="{sid}"[^>]*>.*?<w:name w:val="([^"]+)"/>', sty, re.S)
    print(f"styleId {sid}: {sm.group(1) if sm else 'NOT FOUND'}")
