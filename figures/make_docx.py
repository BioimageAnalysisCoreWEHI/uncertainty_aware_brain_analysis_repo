"""Minimal stdlib-only converter from a lightly marked-up text file to .docx (no python-docx needed).

Markup (one paragraph per line; blank lines ignored):
  # / ## / ###      heading levels 1-3
  - text           bullet
  [NEW] text       whole paragraph in blue (new section/text to insert)
  [FLAG] text      whole paragraph in red italics (review note; not manuscript text)
  [CHG] text       whole paragraph in orange (suggested replacement for existing text)
  [KEY] text       whole paragraph in grey (colour key / document notes)
  Prefixes combine with headings/bullets, e.g. "[NEW] ## Heading", "[NEW] - bullet".
Inline: **bold**, {{CONFIRM: ...}} -> yellow highlight.

Usage: python make_docx.py input.txt output.docx
"""
import re
import sys
import zipfile
from xml.sax.saxutils import escape

COL = {"NEW": "0070C0", "FLAG": "C00000", "CHG": "C55A11", "KEY": "7F7F7F"}
SIZE = {1: 32, 2: 26, 3: 23}  # half-points


def runs(text, colour=None, italic=False, bold=False, size=None):
    out = []
    for part in re.split(r"(\*\*.+?\*\*|\{\{.+?\}\})", text):
        if not part:
            continue
        b, hl, t = bold, False, part
        if part.startswith("**") and part.endswith("**"):
            b, t = True, part[2:-2]
        elif part.startswith("{{") and part.endswith("}}"):
            hl, t = True, part[2:-2]
        rpr = '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>'
        if b:
            rpr += "<w:b/>"
        if italic:
            rpr += "<w:i/>"
        if colour:
            rpr += f'<w:color w:val="{colour}"/>'
        if size:
            rpr += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        if hl:
            rpr += '<w:highlight w:val="yellow"/>'
        out.append(f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escape(t)}</w:t></w:r>')
    return "".join(out)


def para(line):
    colour, italic = None, False
    m = re.match(r"\[(NEW|FLAG|CHG|KEY)\]\s*", line)
    tag = None
    if m:
        tag = m.group(1)
        colour = COL[tag]
        italic = tag == "FLAG"
        line = line[m.end():]
    ppr = '<w:spacing w:after="120" w:line="276" w:lineRule="auto"/>'
    h = re.match(r"(#{1,3})\s+", line)
    if h:
        lvl = len(h.group(1))
        line = line[h.end():]
        ppr = f'<w:keepNext/><w:spacing w:before="{240 if lvl == 1 else 180}" w:after="80"/>'
        return f"<w:p><w:pPr>{ppr}</w:pPr>{runs(line, colour, italic, True, SIZE[lvl])}</w:p>"
    if line.startswith("- "):
        line = "•\t" + line[2:]
        ppr = ('<w:tabs><w:tab w:val="left" w:pos="567"/></w:tabs><w:spacing w:after="60"/>'
               '<w:ind w:left="567" w:hanging="283"/>')
    if tag == "FLAG":
        ppr += ('<w:pBdr><w:left w:val="single" w:sz="18" w:space="6" w:color="C00000"/></w:pBdr>'
                '<w:shd w:val="clear" w:color="auto" w:fill="FDECEC"/>')
    return f"<w:p><w:pPr>{ppr}</w:pPr>{runs(line, colour, italic, False, 22)}</w:p>"


IMAGES = []  # (rid, zip name, source path)


def image_para(spec):
    """[IMG] path | max_width_in | max_height_in  -> inline picture, aspect kept, never upscaled past 600 dpi size."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    parts = [p.strip() for p in spec.split("|")]
    path = parts[0]
    mw = float(parts[1]) if len(parts) > 1 else 6.69
    mh = float(parts[2]) if len(parts) > 2 else 9.3
    with Image.open(path) as im:
        px, py = im.size
    w_in, h_in = px / 600, py / 600                     # native size at 600 dpi
    s = min(1.0, mw / w_in, mh / h_in)                 # only ever shrink (raises effective dpi)
    cx, cy = int(w_in * s * 914400), int(h_in * s * 914400)
    n = len(IMAGES) + 1
    rid, name = f"rIdImg{n}", f"media/image{n}.png"
    IMAGES.append((rid, name, path))
    return ('<w:p><w:pPr><w:keepNext/><w:jc w:val="center"/><w:spacing w:before="120" w:after="120"/></w:pPr><w:r><w:drawing>'
            '<wp:inline distT="0" distB="0" distL="0" distR="0" '
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{n}" name="Picture {n}"/>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="{n}" name="image{n}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>'
            '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
            '</wp:inline></w:drawing></w:r></w:p>')


def table_xml(rows):
    """rows: list of lists of cell strings; first row bold header."""
    out = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>'
           '<w:tblBorders><w:top w:val="single" w:sz="6"/><w:bottom w:val="single" w:sz="6"/>'
           '<w:insideH w:val="single" w:sz="2" w:color="BFBFBF"/></w:tblBorders></w:tblPr>']
    for i, r in enumerate(rows):
        out.append("<w:tr>")
        for c in r:
            out.append('<w:tc><w:tcPr><w:tcMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tcMar></w:tcPr>'
                       f'<w:p><w:pPr><w:spacing w:after="0"/></w:pPr>{runs(c, None, False, i == 0, 18)}</w:p></w:tc>')
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def build(src, dst):
    lines = [l.rstrip("\n") for l in open(src, encoding="utf-8")]
    chunks, tbl = [], []
    for l in lines + [""]:
        if l.startswith("|"):
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                tbl.append(cells)
            continue
        if tbl:
            chunks.append(table_xml(tbl) + '<w:p><w:pPr><w:spacing w:after="60"/></w:pPr></w:p>')
            tbl = []
        if not l.strip():
            continue
        if l.startswith("[PAGEBREAK]"):
            chunks.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
        elif l.startswith("[IMG]"):
            chunks.append(image_para(l[5:].strip()))
        else:
            chunks.append(para(l))
    body = "".join(chunks)
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           f'<w:body>{body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
           '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="567" w:footer="567" w:gutter="0"/>'
           '</w:sectPr></w:body></w:document>')
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Default Extension="png" ContentType="image/png"/>'
          '<Override PartName="/word/document.xml" '
          'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/></Relationships>')
    doc_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                + "".join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                          f'Target="{name}"/>' for rid, name, _ in IMAGES)
                + '</Relationships>')
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", doc)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        for _, name, path in IMAGES:                     # original 600 dpi PNG bytes, unmodified
            z.write(path, "word/" + name, compress_type=zipfile.ZIP_STORED)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
    print("wrote", sys.argv[2])
