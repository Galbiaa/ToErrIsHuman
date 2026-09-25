"""Dump numbering summary + figure layout from report_consegna.docx."""
import zipfile
import xml.etree.ElementTree as ET

DOCX = r"C:\Users\feede\Desktop\Rapporto.docx"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
z = zipfile.ZipFile(DOCX)

# --- numbering.xml ---
root = ET.fromstring(z.read("word/numbering.xml"))
abs_map = {}
for ab in root.findall(f"{W}abstractNum"):
    aid = ab.get(f"{W}abstractNumId")
    linked = ab.find(f"{W}styleLink")
    info = []
    for lvl in ab.findall(f"{W}lvl"):
        ilvl = lvl.get(f"{W}ilvl")
        fmt = lvl.find(f"{W}numFmt")
        txt = lvl.find(f"{W}lvlText")
        info.append((ilvl,
                     fmt.get(f"{W}val") if fmt is not None else None,
                     txt.get(f"{W}val") if txt is not None else None))
    abs_map[aid] = (linked.get(f"{W}val") if linked is not None else None, info)
print("== abstractNum ==")
for aid, (link, info) in abs_map.items():
    print(f"abs {aid} styleLink={link} lvl0={info[0] if info else None} nlevels={len(info)}")
print("== num ==")
for num in root.findall(f"{W}num"):
    nid_el = num.find(f"{W}numId")
    nid = nid_el.get(f"{W}val") if nid_el is not None else num.get(f"{W}numId")
    aid_el = num.find(f"{W}abstractNumId")
    aid = aid_el.get(f"{W}val") if aid_el is not None else num.get(f"{W}abstractNumId")
    override = len(num.findall(f"{W}lvlOverride"))
    print(f"numId {nid} -> abs {aid} overrides={override}")

# --- figure layout in document.xml ---
doc = ET.fromstring(z.read("word/document.xml"))
body = doc.find(f"{W}body")
paras = list(body)
print("== drawings context ==")
for i, p in enumerate(paras):
    if p.tag != f"{W}p":
        continue
    blips = [b.get(f"{R}embed") for b in p.iter(f"{A}blip")]
    if not blips:
        continue
    ppr = p.find(f"{W}pPr")
    st = ppr.find(f"{W}pStyle").get(f"{W}val") if ppr is not None and ppr.find(f"{W}pStyle") is not None else None
    texts = "".join(t.text or "" for t in p.iter(f"{W}t"))
    print(f"[{i}] pStyle={st} embed={blips} text={texts[:60]!r}")
    for j in list(range(max(0, i - 2), i)) + list(range(i + 1, min(i + 3, len(paras)))):
        q = paras[j]
        if q.tag != f"{W}p":
            print(f"    [{j}] <{q.tag.split('}')[1]}>")
            continue
        qppr = q.find(f"{W}pPr")
        qst = qppr.find(f"{W}pStyle").get(f"{W}val") if qppr is not None and qppr.find(f"{W}pStyle") is not None else None
        qblips = [b.get(f"{R}embed") for b in q.iter(f"{A}blip")]
        qtexts = "".join(t.text or "" for t in q.iter(f"{W}t"))
        mark = "PRE" if j < i else "nxt"
        print(f"    {mark}[{j}] pStyle={qst} embed={qblips} text={qtexts[:80]!r}")

# --- raw XML snippets ---
def snip(idx, n=2600):
    p = paras[idx]
    x = ET.tostring(p, encoding="unicode")
    print(f"\n--- para[{idx}] len={len(x)} ---")
    print(x[:n].encode("ascii", "replace").decode())

print("\n== first 14 paragraphs ==")
for i, p in enumerate(paras[:14]):
    if p.tag != f"{W}p":
        print(f"[{i}] <{p.tag.split('}')[1]}>")
        continue
    ppr = p.find(f"{W}pPr")
    st = ppr.find(f"{W}pStyle").get(f"{W}val") if ppr is not None and ppr.find(f"{W}pStyle") is not None else None
    texts = "".join(t.text or "" for t in p.iter(f"{W}t"))
    print(f"[{i}] pStyle={st} text={texts[:80].encode('ascii','replace').decode()!r}")

print("\n== raw XML samples ==")
snip(36, 1600)   # caption 905
snip(37, 3000)   # drawing para 866
snip(38, 800)    # empty 907
snip(0, 1200)    # title?

# first table
tbl = body.find(f"{W}tbl")
if tbl is not None:
    x = ET.tostring(tbl, encoding="unicode")
    print(f"\n--- first tbl len={len(x)} ---")
    print(x[:3200].encode("ascii", "replace").decode())

# a code paragraph (pStyle 914)
for i, p in enumerate(paras):
    if p.tag != f"{W}p":
        continue
    ppr = p.find(f"{W}pPr")
    st = ppr.find(f"{W}pStyle").get(f"{W}val") if ppr is not None and ppr.find(f"{W}pStyle") is not None else None
    if st == "914":
        snip(i, 1600)
        break

# a 881 paragraph
for i, p in enumerate(paras):
    if p.tag != f"{W}p":
        continue
    ppr = p.find(f"{W}pPr")
    st = ppr.find(f"{W}pStyle").get(f"{W}val") if ppr is not None and ppr.find(f"{W}pStyle") is not None else None
    if st == "881":
        snip(i, 1200)
        break

# numbered (numId 2) and bullet (numId 3) pPr
for want in ("2", "3"):
    for p in paras:
        if p.tag != f"{W}p":
            continue
        ppr = p.find(f"{W}pPr")
        if ppr is None:
            continue
        npr = ppr.find(f"{W}numPr")
        if npr is None:
            continue
        nid = npr.find(f"{W}numId")
        if nid is None or nid.get(f"{W}val") != want:
            continue
        snip(0, 0) if False else None
        x = ET.tostring(ppr, encoding="unicode")
        texts = "".join(t.text or "" for t in p.iter(f"{W}t"))
        print(f"\n--- numPr numId={want} pPr ---")
        print(x.encode("ascii", "replace").decode())
        print("text:", texts[:80].encode("ascii", "replace").decode())
        break

# numId 2 lvlOverride level 0
num_root = ET.fromstring(z.read("word/numbering.xml"))
for num in num_root.findall(f"{W}num"):
    nid_el = num.find(f"{W}numId")
    if nid_el is not None and nid_el.get(f"{W}val") == "2":
        x = ET.tostring(num, encoding="unicode")
        print(f"\n--- num numId=2 (len={len(x)}) ---")
        print(x[:1500].encode("ascii", "replace").decode())
        break

