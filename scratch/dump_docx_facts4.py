"""Round 3: samples of styles 907, 905, 881, 914 + rest of md scan (utf-8 stdout)."""
import re
import sys
import zipfile
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = r"C:\Users\feede\Desktop\Rapporto.docx"
MD = r"C:\Users\feede\Desktop\Uni\ToErrIsHuman\Rapporto_finale.md"

z = zipfile.ZipFile(SRC)
doc = z.read("word/document.xml").decode("utf-8")
sty = z.read("word/styles.xml").decode("utf-8")


def show(sid, n=1, maxlen=1400):
    pat = rf'<w:p><w:pPr><w:pStyle w:val="{sid}"\s*/>(?:(?!</w:p>).)*</w:p>'
    ms = list(re.finditer(pat, doc, re.S))
    print(f"\n--- style {sid}: {len(ms)} paragraphs, first sample ---")
    for m in ms[:n]:
        frag = re.sub(r"(<mc:AlternateContent>).*?(</mc:AlternateContent>)", r"\1...DRAWING...\2", m.group(0), flags=re.S)
        frag = re.sub(r"(<w:drawing>).*?(</w:drawing>)", r"\1...DRAWING...\2", frag, flags=re.S)
        print(frag[:maxlen])


show("907")
show("905", n=2)
show("881")
show("914")

sm = re.search(r'<w:style w:type="[^"]*"[^>]*w:styleId="881"[^>]*>.*?<w:name w:val="([^"]+)"/>', sty, re.S)
print("\nstyle 881 name:", sm.group(1) if sm else "NOT FOUND")

# ================= md scan rest =================
lines = open(MD, encoding="utf-8").read().splitlines()

print("\n--- md bullet lines with context ---")
for i, ln in enumerate(lines, 1):
    if ln.startswith("- "):
        print(f"{i}: {ln[:100]}")

print("\n--- md hr contexts ---")
for i, ln in enumerate(lines, 1):
    if ln.strip() == "---":
        prev = lines[i - 2].strip()[:55] if i >= 2 else ""
        nxt = lines[i].strip()[:55] if i < len(lines) else ""
        print(f"hr@{i}: prev='{prev}' next='{nxt}'")

print("\n--- md exotic lines ---")
for i, ln in enumerate(lines, 1):
    s = ln.strip()
    if s.startswith(">") or s.startswith("![") or re.match(r"\[.+\]\(.+\)", s):
        print(f"{i}: {s[:110]}")
    if re.match(r"^\s{2,}[-*] ", ln):
        print(f"{i}: NESTED-BULLET: {s[:80]}")

print("\n--- md: all single-asterisk italic occurrences ---")
cnt = 0
for i, ln in enumerate(lines, 1):
    s = ln.strip()
    if s.startswith("```"):
        continue
    for m in re.finditer(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", s):
        cnt += 1
        if cnt <= 25:
            print(f"{i}: *{m.group(1)}*")
print("italic total:", cnt)

print("\n--- md: backtick spans containing pipes ---")
for i, ln in enumerate(lines, 1):
    for m in re.finditer(r"`([^`]+)`", ln):
        if "|" in m.group(1):
            print(f"{i}: code-with-pipe: {m.group(1)}")

print("\n--- md: table cell non-ascii chars ---")
weird = Counter()
for i, ln in enumerate(lines, 1):
    if ln.lstrip().startswith("|"):
        for ch in ln:
            if ord(ch) > 127:
                weird[ch] += 1
print(dict(weird))

print("\n--- md: lines 160-176 ---")
for i in range(159, 176):
    print(f"{i+1}: {lines[i][:90]}")

print("\n--- md: lines 583-622 (end of file) ---")
for i in range(582, len(lines)):
    print(f"{i+1}: {lines[i][:110]}")
