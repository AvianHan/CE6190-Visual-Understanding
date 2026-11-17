#!/usr/bin/env python3
"""Build, lint, and render the CE6190 Individual report without Git actions."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
PDF_NAME = "CE6190_ICS_Report_20261117.pdf"


def run(command, *, cwd=PAPER, env=None, log=None, timeout=180):
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    if log is not None:
        Path(log).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {command[0]}\n"
                           + (result.stdout + result.stderr)[-6000:])
    return result.stdout


def tex_environment(build):
    """Use installed TeX; repair only local search/cache when system ls-R is absent."""
    env = dict(os.environ)
    runtime = build / "tex_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["kpsewhich", "article.cls"], capture_output=True, text=True, env=env)
    if result.returncode:
        tree = Path("/usr/share/texlive/texmf-dist")
        if not (tree / "tex/latex/base/article.cls").is_file():
            raise RuntimeError("A complete TeX Live installation is required")
        # No global TeX database or system configuration is changed.
        env["TEXMF"] = str(tree)
        env["TEXMFVAR"] = str(runtime)
        env["TEXMFCONFIG"] = str(runtime / "config")
    run(["kpsewhich", "article.cls"], env=env)
    mapping = subprocess.run(["kpsewhich", "pdftex.map"], capture_output=True, text=True, env=env)
    if mapping.returncode:
        map_names = ["ps2pk35.map", "cm.map", "cmextra.map", "latxfont.map", "symbols.map",
                     "utm.map", "ucr.map", "uhv.map", "usy.map", "uzc.map", "uzd.map"]
        records = {}
        for name in map_names:
            path = Path(run(["kpsewhich", name], env=env).strip())
            for line in path.read_text().splitlines():
                clean = line.strip()
                if clean and not clean.startswith(("%", "#")):
                    records.setdefault(clean.split()[0], clean)
        (runtime / "pdftex.map").write_text("\n".join(records.values()) + "\n")
        env["TEXFONTMAPS"] = str(runtime) + "//:"
    fmt = subprocess.run(["kpsewhich", "-format=fmt", "pdflatex.fmt"],
                         capture_output=True, text=True, env=env)
    if fmt.returncode:
        format_path = runtime / "pdflatex.fmt"
        if not format_path.exists():
            run(["pdftex", "-ini", "-etex", "-progname=pdflatex", "-jobname=pdflatex",
                 "-interaction=nonstopmode", "\\input pdflatex.ini"],
                cwd=runtime, env=env, log=runtime / "format_build_output.txt")
        if not format_path.is_file():
            raise RuntimeError("Local pdflatex format was not created")
    else:
        format_path = Path(fmt.stdout.strip())
    return env, format_path


def contact_sheet(paths, destination):
    from PIL import Image, ImageDraw
    width, height = 336, 458
    columns = 3
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "#dedede")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        page = Image.open(path).convert("RGB")
        page.thumbnail((width - 20, height - 30), Image.Resampling.LANCZOS)
        x = (index % columns) * width + (width - page.width) // 2
        y = (index // columns) * height + 6
        sheet.paste(page, (x, y))
        draw.text(((index % columns) * width + 10, (index // columns) * height + height - 20),
                  f"Page {index + 1}", fill="black")
    sheet.save(destination)


def build(render=True, check_deps=False):
    build_dir = ROOT / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    env, fmt = tex_environment(build_dir)
    if check_deps:
        return {"status": "DEPENDENCIES_READY", "format": str(fmt)}
    if not (PAPER / "body.tex").is_file():
        raise ValueError("body.tex is not ready")
    compile_command = ["pdflatex", f"-fmt={fmt}", "-no-shell-escape", "-interaction=nonstopmode",
                       "-halt-on-error", "-file-line-error", f"-output-directory={build_dir}", "main.tex"]
    run(compile_command, env=env, log=build_dir / "pdflatex_pass1.txt")
    aux = (build_dir / "main.aux").read_text()
    if "\\bibdata" in aux:
        run(["bibtex", "main"], cwd=build_dir, env=env, log=build_dir / "bibtex_output.txt")
    run(compile_command, env=env, log=build_dir / "pdflatex_pass2.txt")
    run(compile_command, env=env, log=build_dir / "pdflatex_pass3.txt")
    if "Label(s) may have changed" in (build_dir / "main.log").read_text(errors="replace"):
        run(compile_command, env=env, log=build_dir / "pdflatex_pass4.txt")
    log = (build_dir / "main.log").read_text(errors="replace")
    warnings = [line for line in log.splitlines() if any(token in line for token in
                ("Overfull ", "Underfull ", "undefined", "multiply defined", "LaTeX Warning:", "Package natbib Warning:"))]
    missing_citations = bool(re.search(r"Citation .* undefined|There were undefined citations|Reference .* undefined|There were undefined references", log))
    if missing_citations:
        raise RuntimeError("Unresolved citation or reference remains; inspect build/main.log")
    pdf = build_dir / "main.pdf"
    deliverables = ROOT / "deliverables"
    deliverables.mkdir(parents=True, exist_ok=True)
    destination = deliverables / PDF_NAME
    shutil.copy2(pdf, destination)
    import pypdf
    reader = pypdf.PdfReader(pdf)
    page_count = len(reader.pages)
    media = [(float(page.mediabox.width), float(page.mediabox.height)) for page in reader.pages]
    if any(abs(w - 612) > 0.1 or abs(h - 792) > 0.1 for w, h in media):
        raise RuntimeError("Unexpected page size: expected US Letter")
    result = {"status": "BUILT_REQUIRES_VISUAL_REVIEW", "pdf": str(destination),
              "pages": page_count, "page_size_pt": [612, 792],
              "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
              "warnings": warnings, "overfull_count": sum(x.startswith("Overfull ") for x in warnings),
              "underfull_count": sum(x.startswith("Underfull ") for x in warnings),
              "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if render:
        render_dir = ROOT / "rendered"
        render_dir.mkdir(parents=True, exist_ok=True)
        # Only files generated by this build are replaced.
        for old in render_dir.glob("page-*.png"):
            old.unlink()
        run(["pdftoppm", "-r", "120", "-png", str(destination), str(render_dir / "page")], timeout=180)
        pages = sorted(render_dir.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
        if len(pages) != page_count:
            raise RuntimeError("Rendered page count does not match PDF")
        contact_sheet(pages, render_dir / "contact_sheet.png")
        result.update(rendered_pages=[str(p) for p in pages], contact_sheet=str(render_dir / "contact_sheet.png"))
    (build_dir / "build_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--check-deps", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(not args.no_render, args.check_deps), indent=2))


if __name__ == "__main__":
    main()
