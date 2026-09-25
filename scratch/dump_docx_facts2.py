"""Dump numbering.xml (via ElementTree) and md structure scan."""
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter

SRC = r"C:\Users\feede\Desktop\Rapporto.docx"
MD = r"C:\Users\feede\Desktop\Uni\ToErrIsHuman\Rapporto_finale.md"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

z = zipfile.ZipFile(SRC)
root = ET.fromstring(z.read("word/numbering.xml"))

abs_fmt = {}
for a in root.findall(f"{W}abstractNum"):
    aid = a.get(f"{W}abstractNumId")
    lvls = []
    for lvl in a.findall(f"{W}lvl"):
        ilvl = lvl.get(f"{W}ilvl")
        fmt = lvl.find(f"{W}numFmt")
        txt = lvl.find(f"{W}lvlText")
        st = lvl.find(f"{W}start")
        if ilvl is not None and fmt is not None:
            lvls.append((ilvl, fmt.get(f"{W}val"),
                         txt.get(f"{W}val") if txt is not None else "",
                         st.get(f"{W}val") if st is not None else ""))
    abs_fmt[aid] = lvls

print("--- num -> abstractNum ---")
for n in root.findall(f"{W}num"):
    nid = n.get(f"{W}numId")
    ref = n.find(f"{W}abstractNumId")
    print(f"numId {nid} -> abstractNum {ref.get(f'{W}val') if ref is not None else '?'}")

print("\n--- abstractNum lvl0/lvl1 formats ---")
for aid, lvls in sorted(abs_fmt.items()):
    desc = "; ".join(f"ilvl{l}: fmt={f} text='{t}' start={s}" for l, f, t, s in lvls[:2])
    print(f"abstractNum {aid}: {desc}")

# --- how the original document uses numPr: style + numId pairs
doc = z.read("word/document.xml").decode("utf-8")
pairs = Counter(re.findall(r'<w:pStyle w:val="(\d+)"\s*/>(?:(?!</w:pPr>).)*?<w:numId w:val="(\d+)"\s*/>', doc, re.S))
print("\n--- (pStyle, numId) pairs in original ---")
print(dict(pairs))

# one full bullet paragraph from the original
bm = re.search(r'<w:p><w:pPr><w:pStyle w:val="868"\s*/>(?:(?!</w:pPr>).)*?<w:numPr>.*?</w:pPr>.{0,400}', doc, re.S)
if bm:
    print("\n--- original bullet paragraph sample ---")
    print(bm.group(0))

# ================= md structure scan =================
print("\n\n========== MD STRUCTURE ==========")
lines = open(MD, encoding="utf-8").read().splitlines()
kinds = Counter()
for i, ln in enumerate(lines, 1):
    s = ln.strip()
    if not s:
        continue
    if s.startswith("```"):
        kinds["fence"] += 1
    elif s.startswith("#"):
        kinds["h" + str(len(s) - len(s.lstrip("#")))] += 1
    elif s.startswith("|"):
        kinds["table_row"] += 1
    elif s in ("---", "***", "___"):
        kinds["hr"] += 1
    elif s.startswith("- ") or s.startswith("* "):
        kinds["bullet"] += 1
    elif re.match(r"^\d+[.)] ", s):
        kinds["numbered"] += 1
    elif re.match(r"^Figura \d+", s):
        kinds["figure_line"] += 1
    else:
        kinds["para"] += 1
print("line kind counts:", dict(kinds))

print("\n--- figure lines ---")
for i, ln in enumerate(lines, 1):
    if re.match(r"^\s*Figura \d+", ln):
        print(f"{i}: {ln}")

print("\n--- numbered list lines ---")
for i, ln in enumerate(lines, 1):
    if re.match(r"^\s*\d+[.)] ", ln):
        print(f"{i}: {ln[:100]}")

print("\n--- code fence languages ---")
for i, ln in enumerate(lines, 1):
    s = ln.strip()
    if s.startswith("```") and len(s) > 3:
        print(f"{i}: {s}")

print("\n--- table separator lines (|---|) with preceding header ---")
for i, ln in enumerate(lines, 1):
    if re.match(r"^\s*\|[\s:-]+\|", ln) and set(ln.replace("|", "").strip()) <= set("-: "):
        print(f"{i}: header='{lines[i-2].strip()[:80]}' sep='{ln.strip()[:60]}'")
