"""Quick inspection of figure structure in original document.xml."""
import re

with open(r"C:\Users\feede/Desktop/Rapporto/word/document.xml", encoding="utf-8") as f:
    content = f.read()

# Show root tag (everything up to the end of the <w:document ...> opening tag)
root_tag_end = content.index("<w:body>")
print("=== ROOT TAG (first 200 chars before <w:body>) ===")
print(repr(content[:root_tag_end]))
print()

# Count body elements
body_start = content.index("<w:body>") + len("<w:body>")
body_end = content.rindex("</w:body>")
body_content = content[body_start:body_end]

# Count element types
p_count = body_content.count("<w:p>")
tbl_count = body_content.count("<w:tbl>")
print(f"Body <w:p> count (approx): {p_count}")
print(f"Body <w:tbl> count (approx): {tbl_count}")

# Find all drawing positions and their containing paragraph context
draws = list(re.finditer(r"<w:drawing>", content))
print(f"\nTotal drawings: {len(draws)}")

for i, m in enumerate(draws):
    start = m.start()
    p_start = content.rfind("<w:p ", 0, start)
    if p_start == -1:
        p_start = content.rfind("<w:p>", 0, start)
    p_end = content.find("</w:p>", start) + len("</w:p>")
    p_xml = content[p_start:p_end]
    
    rids = re.findall(r'r:embed="(rId\d+)"', p_xml)
    texts = re.findall(r'<w:t[^>]*>(.*?)</w:t>', p_xml, re.S)
    
    print(f"\nDrawing {i+1}: r:embed={rids}")
    print(f"  Text in para: {texts}")

# Show a sample table structure
print("\n=== SAMPLE TABLE (first 1500 chars) ===")
tbl_start = content.find("<w:tbl>")
tbl_end = content.find("</w:tbl>", tbl_start) + len("</w:tbl>")
print(content[tbl_start:tbl_end][:1500])

# Show sectPr
print("\n=== SECTPR ===")
sectpr_start = content.rfind("<w:sectPr")
sectpr_end = content.rfind("</w:sectPr>") + len("</w:sectPr>")
print(content[sectpr_start:sectpr_end])



