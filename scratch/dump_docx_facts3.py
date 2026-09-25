"""Round 2: figure/code/heading/sectPr details + FIG_RID check + md heading/italic scan."""
import re
import zipfile
import xml.etree.ElementTree as ET

SRC = r"C:\Users\feede\Desktop\Rapporto.docx"
MD = r"C:\Users\feede\Desktop\Uni\ToErrIsHuman\Rapporto_finale.md"

z = zipfile.ZipFile(SRC)
doc = z.read("word/document.xml").decode("utf-8")

# --- pStyle counts (tolerant of " />")
counts = re.findall(r'<w:pStyle w:val="(\d+)"\s*/>', doc)
from collections import Counter
print("--- pStyle counts ---", dict(Counter(counts)))

# --- full figure paragraph (first one, complete)
m = re.search(r'<w:p><w:pPr><w:pStyle w:val="866"\s*/>(?:(?!</w:p>).)*?<w:drawing>(?:(?!</w:p>).)*</w:p>', doc, re.S)
if m:
    frag = m.group(0)
    print("\n--- full figure paragraph, structure (tags only, truncated 1800) ---")
    # strip the drawing innards for readability
    frag_s = re.sub(r"(<mc:AlternateContent>).*?(</mc:AlternateContent>)", r"\1...DRAWING...\2", frag, flags=re.S)
    print(frag_s[:1800])

# --- code block sample: find paragraphs with pStyle 914 and show two consecutive ones raw
cmm = re.search(r'(<w:p><w:pPr><w:pStyle w:val="914"\s*/>.*?</w:p>\s*<w:p><w:pPr><w:pStyle w:val="914"\s*/>.*?</w:p>)', doc, re.S)
print("\n--- code paragraphs sample ---")
print(cmm.group(1)[:1500] if cmm else "none found")

# --- heading full paragraphs (878, 879, 880) — first of each, tags only
for sid in ("878", "879", "880"):
    hm = re.search(rf'<w:p><w:pPr><w:pStyle w:val="{sid}"\s*/>.*?</w:p>', doc, re.S)
    if hm:
        frag = re.sub(r"<w:t[^>]*>.*?</w:t>", "<T/>", hm.group(0), flags=re.S)
        print(f"\n--- heading {sid} paragraph ---")
        print(frag[:800])

# --- does 867 appear? sample
p867 = re.search(r'<w:p><w:pPr><w:pStyle w:val="867"\s*/>.*?</w:p>', doc, re.S)
if p867:
    frag = re.sub(r"<w:t[^>]*>.*?</w:t>", "<T/>", p867.group(0), flags=re.S)
    print("\n--- first-paragraph 867 sample ---")
    print(frag[:600])
else:
    print("\n(no 867 in original)")

# --- sectPr
sp = re.search(r"<w:sectPr\b.*</w:sectPr>", doc, re.S)
print("\n--- sectPr ---")
print(sp.group(0) if sp else "none")

# --- FIG_RID mapping: print each drawing's docPr descr + first r:embed
print("\n--- drawings: descr -> rId ---")
pos = 0
while True:
    s = doc.find("<w:drawing>", pos)
    if s == -1:
        break
    e = doc.find("</w:drawing>", s)
    frag = doc[s:e]
    rid = re.search(r'r:embed="(rId\d+)"', frag)
    desc = re.search(r'<wp:docPr[^>]*descr="([^"]*)"', frag)
    name = re.search(r'<wp:docPr[^>]*name="([^"]*)"', frag)
    print(f"{rid.group(1) if rid else '?':8s} | descr={desc.group(1) if desc else '-'}")
    pos = s + 1

# ================= md scan =================
lines = open(MD, encoding="utf-8").read().splitlines()
print("\n--- md headings ---")
for i, ln in enumerate(lines, 1):
    if ln.startswith("#"):
        print(f"{i}: {ln}")

print("\n--- md single-asterisk italic lines ---")
n = 0
for i, ln in enumerate(lines, 1):
    s = ln.strip()
    if s.startswith("```"):
        continue
    # single asterisks not part of **
    if re.search(r"(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)", s):
        # ignore lines that are pure bullets
        n += 1
        print(f"{i}: {s[:110]}")
print("count:", n)

print("\n--- md exotic lines (> quote, image, link, nested bullets) ---")
for i, ln in enumerate(lines, 1):
    s = ln.strip()
    if s.startswith(">") or s.startswith("![") or re.match(r"\[.+\]\(.+\)", s):
        print(f"{i}: {s[:110]}")
    if re.match(r"^\s{2,}[-*] ", ln):
        print(f"{i}: NESTED-BULLET: {s[:80]}")

print("\n--- md bullet lines ---")
for i, ln in enumerate(lines, 1):
    if ln.startswith("- "):
        print(f"{i}: {ln[:100]}")

print("\n--- lines around each hr ---")
for i, ln in enumerate(lines, 1):
    if ln.strip() == "---":
        print(f"hr@{i}: prev='{lines[i-2].strip()[:60]}' next='{lines[i].strip()[:60] if i < len(lines) else ''}'")
