"""Build Rapporto_finale.docx from Rapporto_finale.md.

Reuses the package of C:\\Users\\feede\\Desktop\\Rapporto.docx: styles, theme,
fonts, numbering, the 9 embedded images (verbatim <w:drawing> XML) and the
section properties. Only word/document.xml is regenerated.

Style map (from the source package):
  878 H1 / 879 H2 / 880 H3 / 881 bold subheading / 866 body / 867 first para
  868 list para / 900 table / 905 caption / 907 figure spacer / 914 code
Numbering: numId 2 = decimal "%1."  (md ordered lists)
           numId 3 = bullet "\\uf0b7" (md "- " lists)
"""
import re
import sys
import zipfile
from pathlib import Path

SRC_DOCX = r"C:\Users\feede\Desktop\Rapporto.docx"
MD = Path(r"c:\Users\feede\Desktop\Uni\ToErrIsHuman\Rapporto_finale.md")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"c:\Users\feede\Desktop\Uni\ToErrIsHuman\Rapporto_finale.docx")

STYLE_H1, STYLE_H2, STYLE_H3 = "878", "879", "880"
STYLE_SUB = "881"      # bold subheading line ("**Etichetta**" alone)
STYLE_BODY = "866"
STYLE_FIRST = "867"
STYLE_LIST = "868"
STYLE_TBL = "900"
STYLE_CAPTION = "905"
STYLE_FIGSP = "907"
STYLE_CODE = "914"
NUM_DECIMAL = "2"
NUM_BULLET = "3"

# figure number -> drawing relationship id (document.xml.rels of the source)
FIG_RID = {
    1: "rId9", 2: "rId10", 3: "rId11", 4: "rId12", 5: "rId13",
    6: "rId14", 7: "rId15", 8: "rId16", 9: "rId17",
}

z = zipfile.ZipFile(SRC_DOCX)
_orig = z.read("word/document.xml").decode("utf-8")

# verbatim <w:drawing> fragments keyed by r:embed id
DRAWINGS = {}
_pos = 0
while True:
    _s = _orig.find("<w:drawing>", _pos)
    if _s == -1:
        break
    _e = _orig.find("</w:drawing>", _s)
    if _e == -1:
        break
    frag = _orig[_s:_e + len("</w:drawing>")]
    m = re.search(r'r:embed="(rId\d+)"', frag)
    if m and m.group(1) not in DRAWINGS:
        DRAWINGS[m.group(1)] = frag
    _pos = _s + 1

m = re.search(r"<w:sectPr\b.*?</w:sectPr>", _orig, flags=re.S)
SECTPR = m.group(0) if m else ""


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def runs(text: str, *, bold_all: bool = False, no_bold: bool = False) -> str:
    """Split **bold** and `code` into runs."""
    out = []
    for part in re.split(r"(\*\*.+?\*\*|`[^`]+`)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            out.append(f'<w:r><w:rPr><w:b/><w:bCs/></w:rPr>'
                       f'<w:t xml:space="preserve">{esc(part[2:-2])}</w:t></w:r>')
        elif part.startswith("`") and part.endswith("`"):
            out.append('<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" '
                       'w:cs="Consolas"/></w:rPr>'
                       f'<w:t xml:space="preserve">{esc(part[1:-1])}</w:t></w:r>')
        else:
            rpr = ('<w:b/><w:bCs/>' if bold_all
                   else '<w:b w:val="0"/><w:bCs w:val="0"/>' if no_bold else '')
            out.append(f'<w:r><w:rPr>{rpr}</w:rPr>'
                       f'<w:t xml:space="preserve">{esc(part)}</w:t></w:r>')
    return "".join(out)


def para(text_runs: str, style: str, *, jc: str | None = None, extra: str = "") -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/>'
    if jc:
        ppr += f'<w:jc w:val="{jc}"/>'
    ppr += extra + "</w:pPr>"
    return f"<w:p>{ppr}{text_runs}</w:p>"


def empty(style: str) -> str:
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr></w:p>'


def code_para(line: str) -> str:
    rpr = '<w:rPr><w:b w:val="0"/><w:bCs w:val="0"/></w:rPr>'
    t = f'<w:t xml:space="preserve">{esc(line)}</w:t>' if line else ""
    return (f'<w:p><w:pPr><w:pStyle w:val="{STYLE_CODE}"/>{rpr}</w:pPr>'
            f'<w:r>{rpr}{t}</w:r></w:p>')


def list_para(text: str, num_id: str) -> str:
    extra = (f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{num_id}"/></w:numPr>'
             '<w:rPr><w:b w:val="0"/><w:bCs w:val="0"/></w:rPr>')
    return para(runs(text), STYLE_LIST, extra=extra)


def figure_block(num: int, caption: str, subtitle: str) -> str:
    draw = DRAWINGS.get(FIG_RID.get(num, ""), "")
    parts = [empty(STYLE_CAPTION)]
    if caption:
        parts.append(para(runs(caption, no_bold=True), STYLE_CAPTION))
    inner = draw + (runs(subtitle, no_bold=True) if subtitle else "")
    parts.append(para(inner, STYLE_BODY, jc="center"))
    parts.append(empty(STYLE_FIGSP))
    parts.append(empty(STYLE_CAPTION))
    return "".join(parts)


def table(rows: list[list[str]]) -> str:
    header, data = rows[0], rows[1:]
    ncol = len(header)
    widths = [max(8, 90 // ncol)] * ncol
    grid = "".join(f'<w:gridCol w:w="{w * 100}"/>' for w in widths)
    xml = [f'<w:tbl><w:tblPr><w:tblStyle w:val="{STYLE_TBL}"/>'
           f'<w:tblW w:w="0" w:type="auto"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>']
    for ri, row in enumerate([header] + data):
        xml.append("<w:tr>")
        for ci in range(ncol):
            cell = row[ci] if ci < len(row) else ""
            bold = ri == 0 and not set(cell) <= set("-: ")
            jc = '<w:jc w:val="right"/>' if "--:" in header[ci] else ""
            xml.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci] * 100}" w:type="dxa"/></w:tcPr>'
                f'<w:p><w:pPr>{jc}</w:pPr>{runs(cell, bold_all=bold)}</w:p></w:tc>')
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


# ---------------------------------------------------------------- md parsing
lines = MD.read_text(encoding="utf-8").splitlines()

body: list[str] = []
i = 0
first_after_heading = False
in_code = False
code_buf: list[str] = []

FIG_TITLE = re.compile(r"^Figura (\d+)\.\s*(.*)$")
FIG_SUB = re.compile(r"^Figura (\d+)\s*-\s*(.*)$")

while i < len(lines):
    ln = lines[i]

    if ln.lstrip().startswith("```"):
        if in_code:
            body.append("".join(code_para(c) for c in code_buf))
            code_buf = []
            in_code = False
        else:
            in_code = True
        i += 1
        continue
    if in_code:
        code_buf.append(ln)
        i += 1
        continue

    s = ln.strip()
    if not s:
        i += 1
        continue

    if s == "---":
        body.append(empty(STYLE_BODY))
        i += 1
        continue

    m = re.match(r"^(#{1,3})\s+(.*)$", s)
    if m:
        style = {1: STYLE_H1, 2: STYLE_H2, 3: STYLE_H3}[len(m.group(1))]
        body.append(para(runs(m.group(2), no_bold=True), style))
        first_after_heading = True
        i += 1
        continue

    mt, ms = FIG_TITLE.match(s), FIG_SUB.match(s)
    if mt or ms:
        num = (mt or ms).group(1)
        caption = mt.group(2).strip() if mt else ""
        subtitle = ""
        j = i + 1
        if mt:
            while j < len(lines):
                s2 = lines[j].strip()
                m2 = FIG_SUB.match(s2)
                if m2 and m2.group(1) == num:
                    subtitle = m2.group(2).strip()
                    j += 1
                    break
                if not s2:
                    j += 1
                    continue
                break
        body.append(figure_block(int(num), f"Figura {num}. {caption}".strip(), subtitle))
        first_after_heading = False
        i = j
        continue

    if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
        rows = [[c.strip() for c in s.strip("|").split("|")]]
        i += 2
        while i < len(lines) and lines[i].strip().startswith("|"):
            rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
            i += 1
        body.append(table(rows))
        body.append(empty(STYLE_BODY))
        first_after_heading = False
        continue

    mo = re.match(r"^(\d+)[.)]\s+(.*)$", s)
    if mo:
        body.append(list_para(mo.group(2), NUM_DECIMAL))
        first_after_heading = False
        i += 1
        continue
    mb = re.match(r"^[-*]\s+(.*)$", s)
    if mb:
        body.append(list_para(mb.group(1), NUM_BULLET))
        first_after_heading = False
        i += 1
        continue

    if re.fullmatch(r"\*\*[^*]+\*\*", s):
        body.append(para(runs(s[2:-2], no_bold=True), STYLE_SUB))
        first_after_heading = False
        i += 1
        continue

    style = STYLE_FIRST if first_after_heading else STYLE_BODY
    body.append(para(runs(s), style))
    first_after_heading = False
    i += 1

if in_code and code_buf:
    body.append("".join(code_para(c) for c in code_buf))

document = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
    'xmlns:o="urn:schemas-microsoft-com:office:office" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
    'xmlns:v="urn:schemas-microsoft-com:vml" '
    'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:w10="urn:schemas-microsoft-com:office:word" '
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
    'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
    'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
    'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
    'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
    'mc:Ignorable="w14 wp14">'
    f"<w:body>{''.join(body)}{SECTPR}</w:body></w:document>"
)

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zo:
    for item in z.infolist():
        data = z.read(item.filename)
        if item.filename == "word/document.xml":
            data = document.encode("utf-8")
        zo.writestr(item, data)

print(f"written {OUT} ({OUT.stat().st_size} bytes), "
      f"{len(body)} block elements, {len(DRAWINGS)} drawings available")
