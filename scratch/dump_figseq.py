import sys, zipfile, re
z = zipfile.ZipFile(sys.argv[1] if len(sys.argv) > 1 else r'c:\Users\feede\Desktop\Uni\ToErrIsHuman\Rapporto_finale.docx')
xml = z.read('word/document.xml').decode('utf-8')
rels = z.read('word/_rels/document.xml.rels').decode('utf-8')
rmap = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
paras = re.findall(r'<w:p[ >].*?</w:p>', xml, re.S)
seq = []
for p in paras:
    txt = ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))
    dr = re.findall(r'r:embed="(rId\d+)"', p)
    st = re.search(r'<w:pStyle w:val="(\d+)"', p)
    st = st.group(1) if st else '-'
    if dr:
        seq.append(('IMG', ','.join(rmap.get(r, '?') for r in dr), st))
    elif txt.strip():
        seq.append(('P', txt.strip(), st))
print('=== sequenza figure nel DOCX (ordine di lettura) ===')
for s in seq:
    if s[0]=='IMG' or re.match(r'Figura\s*\d', s[1]):
        print(s[0], '| style', s[2], '|', s[1][:80])
print('--- paragrafi che iniziano con --- :', [s[1][:40] for s in seq if s[1].startswith('---')])
mod = [s for s in seq if 'Moduli di supporto' in s[1]]
print('--- paragrafo Moduli di supporto: style =', mod[0][2] if mod else 'NON TROVATO')
print('--- paragrafi in stile codice (914):', sum(1 for s in seq if s[2]=='914'))
