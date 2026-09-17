# -*- coding: utf-8 -*-
"""Bauskript fuer das Paper dieses Repos.

    python build.py testbuild          baut in einen Scratch-Ordner, meldet den Fingerabdruck
    python build.py fingerprint [f]    schreibt den Fingerabdruck als JSON (Standard: .fingerprint.json)
    python build.py check [f]          baut und vergleicht gegen einen gespeicherten Fingerabdruck
    python build.py release <ordner>   baut ein flaches Einreichungspaket (EINE .tex,
                                       Abbildungen, PDF, MANIFEST) und prueft es,
                                       indem es das Paket noch einmal baut

WARUM ES DIESES SKRIPT GIBT
---------------------------
`\\graphicspath{{./}{./figures/}}` im Paper ist relativ zum ARBEITSVERZEICHNIS von pdflatex,
nicht zum Ort der .tex. Wer in `paper/` geht und `pdflatex paper.tex` sagt — der
naheliegendste Befehl —, bekommt einen roten Bau mit 9 nicht gefundenen Abbildungen und
keinen Hinweis, woran es liegt. Gebaut werden muss aus dem REPO-WURZELVERZEICHNIS.
Diese eine Zeile Wissen stand nirgends; jetzt steht sie hier und wird erzwungen.

Der FINGERABDRUCK (Seiten · jedes \\label mit seiner Nummer · Literaturstellen · undefinierte
Verweise · fehlende Abbildungen) ist die Gegenprobe fuer strukturelle Eingriffe: wer die .tex
zerlegt, verschiebt oder umsortiert, muss danach denselben Fingerabdruck haben. `check`
vergleicht und gibt einen Fehlercode zurueck, wenn nicht.
"""
import glob
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys

# ------------------------------------------------------------------ Konfiguration
MAIN = 'paper/analytic_prime_indicator'          # ohne .tex, relativ zur Repo-Wurzel
FIGDIR = 'figures'
REPO = os.path.dirname(os.path.abspath(__file__))
PDFLATEX = os.environ.get('PDFLATEX', 'pdflatex')
RUNS = 3

RE_INC = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')
RE_INPUT = re.compile(r'\\(?:input|include)\{([^}]+)\}')


def _sources():
    """Die .tex-Hauptdatei plus alle per \\input eingebundenen Teile."""
    out, queue = [], [MAIN + '.tex']
    seen = set()
    while queue:
        rel = queue.pop(0)
        if rel in seen:
            continue
        p = os.path.join(REPO, rel)
        if not os.path.exists(p):
            p = os.path.join(REPO, rel + '.tex')
            rel = rel + '.tex'
            if not os.path.exists(p):
                continue
        seen.add(rel)
        out.append(rel)
        s = io.open(p, encoding='utf-8').read()
        base = os.path.dirname(rel)
        for inc in RE_INPUT.findall(s):
            cand = inc if inc.endswith('.tex') else inc + '.tex'
            for c in (cand, os.path.join(base, cand)):
                if os.path.exists(os.path.join(REPO, c)):
                    queue.append(c)
                    break
    return out


def _figures():
    need = set()
    for rel in _sources():
        need |= set(RE_INC.findall(io.open(os.path.join(REPO, rel), encoding='utf-8').read()))
    return sorted(need)



def inline(rel, _depth=0):
    """\\input-Teile wieder einsetzen — ein Einreichungspaket ist EINE .tex.

    Die Zerlegung in `paper/parts/` ist ein Arbeitsmittel des Repos. Nach aussen soll
    dieselbe einzelne Datei gehen wie bisher; das haelt die Pakete vergleichbar und
    nimmt jede Unsicherheit darueber, ob der Empfaenger `\\input` aufloest.
    """
    if _depth > 8:
        raise RuntimeError('input-Schachtelung zu tief: ' + rel)
    p = os.path.join(REPO, rel)
    if not os.path.exists(p):
        p = os.path.join(REPO, rel + '.tex')
    txt = io.open(p, encoding='utf-8').read()
    base = os.path.dirname(rel)

    def sub(m):
        inc = m.group(1)
        cand = inc if inc.endswith('.tex') else inc + '.tex'
        for c in (cand, os.path.join(base, cand)):
            if os.path.exists(os.path.join(REPO, c)):
                return inline(c, _depth + 1)
        return m.group(0)

    return RE_INPUT.sub(sub, txt)


def build(outdir):
    """Baut aus der Repo-Wurzel in outdir. Das Repo bleibt sauber (-output-directory)."""
    os.makedirs(outdir, exist_ok=True)
    name = os.path.basename(MAIN)
    rc = []
    for _ in range(RUNS):
        p = subprocess.run([PDFLATEX, '-interaction=nonstopmode',
                            '-output-directory=' + os.path.abspath(outdir),
                            MAIN + '.tex'],
                           cwd=REPO, capture_output=True, text=True, errors='replace')
        rc.append(p.returncode)
    logp = os.path.join(outdir, name + '.log')
    if not os.path.exists(logp):
        print('FEHLER: kein Log erzeugt. Ist pdflatex im PATH? (PDFLATEX=... setzen)')
        sys.exit(2)
    log = io.open(logp, encoding='utf-8', errors='replace').read()
    aux = io.open(os.path.join(outdir, name + '.aux'), encoding='utf-8',
                  errors='replace').read()
    # LaTeX bricht Logzeilen um — deshalb erst Zeilenumbrueche entfernen
    flat = log.replace('\n', '')
    m = re.search(r'Output written on .*?\((\d+) pages', flat)
    labels = {a: b for a, b, _ in
              re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}', aux)}
    pdf = os.path.join(outdir, name + '.pdf')
    return dict(
        pages=int(m.group(1)) if m else None,
        labels=labels,
        n_labels=len(labels),
        bibcite=len(re.findall(r'\\bibcite\{', aux)),
        undef=len(re.findall(r'Warning: (?:Reference|Citation) `', log)),
        missing=sorted(set(re.findall(r"File `([^']+)' not found", log))),
        errors=sorted(set(re.findall(r'^! (.+)$', log, re.M)))[:10],
        pdf=pdf if os.path.exists(pdf) else None,
        sources=_sources(),
    )


def show(fp):
    print('  Seiten                 %s' % fp['pages'])
    print('  \\label gesamt          %d' % fp['n_labels'])
    print('  Literaturstellen       %d' % fp['bibcite'])
    print('  undefinierte Verweise  %d' % fp['undef'])
    print('  fehlende Abbildungen   %d %s' % (len(fp['missing']), fp['missing'] or ''))
    print('  Quelldateien           %d (%s)' % (len(fp['sources']),
                                                ', '.join(fp['sources'][:4])
                                                + (' …' if len(fp['sources']) > 4 else '')))
    if fp['errors']:
        print('  LaTeX-Fehler:')
        for e in fp['errors']:
            print('     ! %s' % e)


def compare(a, b):
    """True, wenn der Bau strukturell unveraendert ist."""
    bad = []
    for k in ('pages', 'n_labels', 'bibcite', 'undef'):
        if a[k] != b[k]:
            bad.append('%s: %s -> %s' % (k, a[k], b[k]))
    neu = sorted(set(b['labels']) - set(a['labels']))
    weg = sorted(set(a['labels']) - set(b['labels']))
    ver = {k: (a['labels'][k], b['labels'][k])
           for k in set(a['labels']) & set(b['labels'])
           if a['labels'][k] != b['labels'][k]}
    if neu:
        bad.append('neue Labels: %s' % neu)
    if weg:
        bad.append('entfallene Labels: %s' % weg)
    if ver:
        bad.append('umnummeriert: %s' % {k: '%s->%s' % v for k, v in ver.items()})
    if bad:
        print('UNTERSCHIEDE:')
        for x in bad:
            print('   * %s' % x)
        return False
    print('IDENTISCH: Seiten, Labels samt Nummern, Literaturstellen, undefinierte Verweise.')
    return True


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'testbuild'
    scratch = os.path.join(REPO, '.build')

    if cmd == 'testbuild':
        fp = build(scratch)
        print('Testbuild (%s)' % scratch)
        show(fp)
        sys.exit(0 if (fp['pages'] and not fp['missing'] and not fp['errors']) else 1)

    if cmd == 'fingerprint':
        out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, '.fingerprint.json')
        fp = build(scratch)
        fp.pop('pdf', None)
        json.dump(fp, io.open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        show(fp)
        print('geschrieben: %s' % out)
        sys.exit(0)

    if cmd == 'check':
        ref = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, '.fingerprint.json')
        if not os.path.exists(ref):
            print('Kein Referenz-Fingerabdruck: %s  (erst "fingerprint" laufen lassen)' % ref)
            sys.exit(2)
        old = json.load(io.open(ref, encoding='utf-8'))
        new = build(scratch)
        show(new)
        sys.exit(0 if compare(old, new) else 1)

    if cmd == 'release':
        if len(sys.argv) < 3:
            print('Aufruf: python build.py release <zielordner>')
            sys.exit(2)
        dest = os.path.abspath(sys.argv[2])
        fp = build(scratch)
        if not fp['pdf'] or fp['missing'] or fp['undef'] or fp['errors']:
            print('ABBRUCH: der Bau ist nicht sauber.')
            show(fp)
            sys.exit(1)
        os.makedirs(dest, exist_ok=True)
        name = os.path.basename(MAIN)
        # arXiv will ein FLACHES Paket: alle .tex zusammengefuegt waere riskant,
        # deshalb die Teile mitkopieren und die \input-Pfade flach umschreiben.
        srcs = _sources()
        io.open(os.path.join(dest, name + '.tex'), 'w', encoding='utf-8',
                newline='').write(inline(MAIN + '.tex'))
        for f in _figures():
            src = os.path.join(REPO, FIGDIR, f)
            if not os.path.exists(src):
                src = os.path.join(REPO, f)
            shutil.copy(src, os.path.join(dest, os.path.basename(f)))
        shutil.copy(fp['pdf'], os.path.join(dest, name + '.pdf'))
        # Herkunftsnachweis: md5 jeder Quelle und jeder Abbildung
        man = {'main': name + '.tex', 'pages': fp['pages'],
               'assembled_from_parts': len(srcs) > 1,
               'labels': fp['n_labels'], 'bibcite': fp['bibcite'],
               'sources': {}, 'figures': {}}
        for rel in srcs:
            man['sources'][os.path.basename(rel)] = hashlib.md5(
                io.open(os.path.join(REPO, rel), 'rb').read()).hexdigest()
        for f in _figures():
            p = os.path.join(dest, os.path.basename(f))
            man['figures'][os.path.basename(f)] = hashlib.md5(
                io.open(p, 'rb').read()).hexdigest()
        # SELBSTPRUEFUNG: das fertige Paket noch einmal bauen, im Paketordner,
        # und gegen den Repo-Bau halten. Ein Paket, das sich nicht selbst baut,
        # ist kein Paket.
        pk = subprocess.run
        for _ in range(RUNS):
            pk([PDFLATEX, '-interaction=nonstopmode', name + '.tex'],
               cwd=dest, capture_output=True, text=True, errors='replace')
        plog = io.open(os.path.join(dest, name + '.log'), encoding='utf-8',
                       errors='replace').read()
        paux = io.open(os.path.join(dest, name + '.aux'), encoding='utf-8',
                       errors='replace').read()
        pm = re.search(r'Output written on .*?\((\d+) pages', plog.replace('\n', ''))
        pfp = dict(pages=int(pm.group(1)) if pm else None,
                   labels={a: b for a, b, _ in re.findall(
                       r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}', paux)},
                   bibcite=len(re.findall(r'\\bibcite\{', paux)),
                   undef=len(re.findall(r'Warning: (?:Reference|Citation) `', plog)))
        pfp['n_labels'] = len(pfp['labels'])
        print('Paket geschrieben: %s' % dest)
        show(fp)
        print('  Abbildungen kopiert    %d' % len(man['figures']))
        print('Selbstpruefung: das Paket noch einmal gebaut, im Paketordner ...')
        same = compare(fp, pfp)
        man['selfcheck'] = dict(ok=bool(same), pages=pfp['pages'],
                                n_labels=pfp['n_labels'], bibcite=pfp['bibcite'],
                                undef=pfp['undef'])
        json.dump(man, io.open(os.path.join(dest, 'MANIFEST.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        # Bauartefakte gehoeren nicht in ein Einreichungspaket
        for ext in ('.aux', '.log', '.out', '.toc', '.fls', '.fdb_latexmk',
                    '.synctex.gz', '.bbl', '.blg'):
            f = os.path.join(dest, name + ext)
            if os.path.exists(f):
                os.remove(f)
        print('  MANIFEST.json mit md5 je Quelle und Abbildung, plus Selbstpruefung')
        print('  Bauartefakte aus dem Paket entfernt')
        sys.exit(0 if same else 1)

    print(__doc__)
    sys.exit(2)


if __name__ == '__main__':
    main()
