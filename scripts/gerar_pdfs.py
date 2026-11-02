"""Converte relatorio.md e memo_professor.md em HTML para imprimir em PDF (conferir o limite de páginas).

Uso: python scripts/gerar_pdfs.py relatorio.md relatorio.html 10.5 1.8
     python scripts/gerar_pdfs.py memo_professor.md memo_professor.html 10 1.5
Depois imprima o HTML em PDF, por exemplo com o Google Chrome:
     google-chrome --headless --no-pdf-header-footer --print-to-pdf=relatorio.pdf relatorio.html
Requer o pacote `markdown` (não está em requirements.txt: não faz parte da classificação).
"""
import sys, markdown, pathlib
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
body=markdown.markdown(src.read_text(encoding='utf-8'), extensions=['tables','fenced_code'])
FS=sys.argv[3] if len(sys.argv)>3 else "10.5"; MG=sys.argv[4] if len(sys.argv)>4 else "1.8"
css="""@page{size:A4;margin:{MG}cm {MG}cm}body{font-family:-apple-system,Helvetica,Arial,sans-serif;font-size:{FS}pt;line-height:1.35;color:#111}
h1{font-size:16pt;margin:0 0 8px}h2{font-size:12.5pt;margin:14px 0 6px;border-bottom:1px solid #ccc;padding-bottom:2px}
table{border-collapse:collapse;width:100%;font-size:9pt;margin:6px 0}th,td{border:1px solid #bbb;padding:3px 5px;vertical-align:top}th{background:#f2f2f2}
code{font-size:9pt;background:#f4f4f4;padding:0 2px}p,li{margin:4px 0}ul,ol{margin:4px 0 4px 18px;padding:0}"""
css=css.replace("{FS}",FS).replace("{MG}",MG)
out.write_text(f"<!doctype html><html lang='pt-BR'><meta charset='utf-8'><title>{src.stem}</title><style>{css}</style><body>{body}</body></html>",encoding='utf-8')
