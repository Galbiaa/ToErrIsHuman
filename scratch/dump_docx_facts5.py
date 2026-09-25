"""Round 5: original doc paragraph sequence + figure context + md raw fence check."""
import re
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = r"C:\Users\feede\Desktop\Rapporto.docx"
MD = r"C:\Users\feede\Desktop\Uni\ToErrIsHuman\Rapporto_finale.md"

z = zipfile.ZipFile(SRC)
doc = z.read("word/document.xml").decode("utf-8")

# --- paragraph sequence: style + preview for every block-level element in body order
body = doc[doc.index("<w:body>") + 8: doc.index("</w:body>")]
print("--- original body sequence (first 120 blocks) ---")
seq = []
pos = 0
idx = 0
for m in re.finditer(r"<w:(p|tbl|sectPr)[ >]", body):
    if m.start() < pos:
        continue
    kind = m.group(1)
    if kind == "p":
        pm = re.match(r"<w:p><w:pPr>(?:(?!</w:p>).)*?</w:p>", body[m.start():], re.S)
        if not pm:
            continue
        frag = pm.group(0)
        sm = re.search(r'<w:pStyle w:val="(\d+)"', frag)
        style = sm.group(1) if sm else "-"
        has_draw = "<w:drawing>" in frag or "AlternateContent" in frag
        texts = " ".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", frag))[:60]
        numm = re.search(r'<w:numId w:val="(\d+)"', frag)
        extra = f" DRAWING" if has_draw else ""
        extra += f" num{numm.group(1)}" if numm else ""
        seq.append(f"[{idx:3d}] p {style:4s}{extra} | {texts}")
        pos = m.start() + len(frag)
    elif kind == "tbl":
        # find matching close
        em = body.find("</w:tbl>", m.start())
        frag = body[m.start():em + 8]
        rows = frag.count("<w:tr>")
        texts = " ".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", frag))[:60]
        seq.append(f"[{idx:3d}] TBL ({rows} rows) | {texts}")
        pos = em + 8
    else:
        seq.append(f"[{idx:3d}] sectPr")
        pos = m.start() + 8
    idx += 1
print("\n".join(seq[:120]))
print(f"... total blocks: {len(seq)}")

# --- figure paragraphs: which style contains each drawing?
print("\n--- drawing container styles ---")
for m in re.finditer(r"<w:drawing>", doc):
    before = doc[:m.start()]
    ps = re.findall(r'<w:pStyle w:val="(\d+)"', before[-3000:])
    print("drawing in pStyle:", ps[-1] if ps else "?")

# --- what follows the 907 paragraphs?
ms = list(re.finditer(r'<w:p><w:pPr><w:pStyle w:val="907"\s*/>(?:(?!</w:p>).)*</w:p>', doc, re.S))
print("\n--- blocks around first 907 paragraph ---")
if ms:
    s = ms[0].start()
    ctx = doc[max(0, s - 1500): s + 2500]
    ctx = re.sub(r"(<mc:AlternateContent>).*?(</mc:AlternateContent>)", r"\1[DRW]\2", ctx, flags=re.S)
    ctx = re.sub(r"<w:t[^>]*>", "<t>", ctx)
    print(ctx[:3000])

# --- style defs 905 / 907 / 914 / 881 (pPr parts)
sty = z.read("word/styles.xml").decode("utf-8")
for sid in ("905", "907", "914", "881", "868", "866", "867"):
    sm = re.search(rf'<w:style w:type="paragraph"[^>]*w:styleId="{sid}"[^>]*>(.*?)</w:style>', sty, re.S)
    if sm:
        body_xml = sm.group(1)
        nm = re.search(r'<w:name w:val="([^"]+)"/>', body_xml)
        pp = re.search(r"<w:pPr>.*?</w:pPr>", body_xml, re.S)
        print(f"\nstyle {sid} ({nm.group(1) if nm else '?'}): {pp.group(0)[:500] if pp else 'no pPr'}")

# ================= md raw check =================
raw = open(MD, encoding="utf-8").read()
lines = raw.splitlines()
print("\n\n--- md raw lines 583-592 (repr) ---")
for i in range(582, 592):
    print(f"{i+1}: {lines[i]!r}")
print("\n--- md raw lines 617-622 (repr) ---")
for i in range(616, len(lines)):
    print(f"{i+1}: {lines[i]!r}")

print("\n--- md: all fence lines with index ---")
for i, ln in enumerate(lines, 1):
    if ln.strip().startswith("```"):
        print(f"{i}: {ln!r}")

print("\n--- md hr contexts ---")
for i, ln in enumerate(lines, 1):
    if ln.strip() == "---":
        prev = lines[i - 2].strip()[:55] if i >= 2 else ""
        nxt = lines[i].strip()[:55] if i < len(lines) else ""
        print(f"hr@{i}: prev='{prev}' next='{nxt}'")

print("\n--- md: backtick spans containing pipes ---")
found = False
for i, ln in enumerate(lines, 1):
    for m in re.finditer(r"`([^`]+)`", ln):
        if "|" in m.group(1):
            found = True
            print(f"{i}: {m.group(1)!r}")
if not found:
    print("(none)")

print("\n--- md exotic lines ---")
for i, ln in enumerate(lines, 1):
    s = ln.strip()
    if s.startswith(">") or s.startswith("![") or re.match(r"\[.+\]\(.+\)", s):
        print(f"{i}: {s[:110]}")
    if re.match(r"^\s{2,}[-*] ", ln):
        print(f"{i}: NESTED-BULLET: {s[:80]}")

print("\n--- md: full-bold lines (h4 candidates) ---")
for i, ln in enumerate(lines, 1):
    s = ln.strip()
    if re.fullmatch(r"\*\*.+\*\*", s):
        print(f"{i}: {s[:90]}")
