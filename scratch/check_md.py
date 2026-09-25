"""Check numbering refs in document.xml body elements."""
import re

with open(r"C:\Users\feede\Desktop\Rapporto\word\numbering.xml", encoding="utf-8") as f:
    num_content = f.read()

# Find all <w:num> elements with their abstractNumId and check format
for num_match in re.finditer(r'<w:num\s+w:numId="(\d+)"[^>]*>(.*?)</w:num>', num_content, re.S):
    num_id = num_match.group(1)
    inner = num_match.group(2)
    abs_id = re.search(r'<w:abstractNumId w:val="(\d+)"', inner)
    if abs_id:
        abs_num = abs_id.group(1)
        # Find the abstractNum and check its format
        abs_pattern = f'<w:abstractNum w:abstractNumId="{abs_num}"'
        abs_start = num_content.find(abs_pattern)
        if abs_start >= 0:
            abs_end = num_content.find("</w:abstractNum>", abs_start)
            abs_inner = num_content[abs_start:abs_end]
            numFmt = re.search(r'<w:numFmt w:val="([^"]+)"', abs_inner)
            lvlText = re.search(r'<w:lvlText w:val="([^"]+)"', abs_inner)
            is_bullet = "bullet" in (numFmt.group(1) if numFmt else "")
            fmt = numFmt.group(1) if numFmt else "??"
            text = lvlText.group(1) if lvlText else "??"
            print(f"numId={num_id} -> abstractNumId={abs_num}, fmt={fmt}, text={text}, bullet={is_bullet}")

# Show the full abstractNum 2 structure
print("\n=== Full abstractNum elements ===")
for abs_match in re.finditer(r'<w:abstractNum\b[^>]*>(.*?)</w:abstractNum>', num_content, re.S):
    abs_full = abs_match.group(0)
    abs_id = re.search(r'abstractNumId="(\d+)"', abs_full)
    if abs_id:
        aid = abs_id.group(1)
        numFmt = re.search(r'numFmt w:val="([^"]+)"', abs_full)
        lvlText = re.search(r'lvlText w:val="([^"]+)"', abs_full)
        fmt = numFmt.group(1) if numFmt else "??"
        text = lvlText.group(1) if lvlText else "??"
        print(f"  abstractNumId={aid}: fmt={fmt}, lvlText={text}")





