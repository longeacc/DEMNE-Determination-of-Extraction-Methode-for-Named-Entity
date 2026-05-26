"""Notebook viewer / executor embarqué dans la GUI Tkinter.

Lit un fichier .ipynb (REST.ipynb par défaut) et reproduit son contenu sous
forme de widgets Tk natifs :
  - cellules markdown → labels formatés
  - cellules code     → Text widget éditable + bouton ▶ Run
  - sorties           → text/plain, image/png, text/html (texte), errors

Exécution 100 % in-process via IPython.core.interactiveshell.InteractiveShell.
**Aucun port ouvert**, aucune connexion réseau, pas de ZMQ ni de kernel séparé.

Limite physique : les ipywidgets interactifs (widgets.Tab, dropdowns, …)
sont rendus en placeholder car ils nécessitent un frontend Comm (HTML/JS).
"""

from __future__ import annotations

import base64
import io
import os
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable

import nbformat
from IPython.core.interactiveshell import InteractiveShell


# ---------------------------------------------------------------------------
class _ScrollFrame(ttk.Frame):
    """Conteneur scrollable contenant un Frame `inner`."""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                 background="#FAFAFA")
        self.vsb = ttk.Scrollbar(self, orient="vertical",
                                 command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner,
                                                   anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        # Wheel scroll
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _on_resize(self, _e):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_resize(self, e):
        self.canvas.itemconfigure(self.inner_id, width=e.width)

    def _on_wheel(self, e):
        try:
            self.canvas.yview_scroll(int(-e.delta / 120), "units")
        except Exception:
            pass


# ---------------------------------------------------------------------------
class NotebookViewer(ttk.Frame):
    """Affiche et exécute un .ipynb sans serveur."""

    def __init__(self, master, nb_path: Path,
                 log_fn: Callable[[str], None] | None = None):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        self.nb_path = Path(nb_path)
        self.log_fn = log_fn or (lambda _m: None)

        # IPython shell — exécution in-process, zéro socket
        InteractiveShell.clear_instance()
        self.shell = InteractiveShell.instance()
        self.shell.user_ns["__file__"] = str(self.nb_path)

        # Insère le répertoire du notebook au sys.path pour reproduire le
        # comportement de Jupyter (sinon `from REST_modules import *` casse).
        nb_dir = str(self.nb_path.parent.resolve())
        if nb_dir not in sys.path:
            sys.path.insert(0, nb_dir)
        self._cell_widgets: list[dict] = []
        # Pour garder une référence aux PhotoImage (sinon GC)
        self._img_refs: list = []

        self._build_toolbar()
        self._build_body()
        self._load_notebook()

    # ------------------------------------------------------------------ UI
    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=6); bar.pack(fill="x")
        ttk.Label(bar, text=self.nb_path.name,
                  font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(bar, text=f" — exécuté in-process (aucun port)",
                  foreground="#666").pack(side="left", padx=4)
        ttk.Button(bar, text="▶ Tout exécuter",
                   command=self.run_all).pack(side="right", padx=4)
        ttk.Button(bar, text="🔄 Recharger",
                   command=self._reload).pack(side="right")
        ttk.Button(bar, text="🧹 Reset kernel",
                   command=self._reset_kernel).pack(side="right", padx=4)
        ttk.Separator(self, orient="horizontal").pack(fill="x")

    def _build_body(self) -> None:
        self.scroll = _ScrollFrame(self)
        self.scroll.pack(fill="both", expand=True)
        self.body = self.scroll.inner

    # ------------------------------------------------------------------ load
    def _load_notebook(self) -> None:
        if not self.nb_path.exists():
            ttk.Label(self.body, text=f"Notebook introuvable : {self.nb_path}",
                       foreground="#B12727").pack(anchor="w", padx=8, pady=8)
            return

        try:
            nb = nbformat.read(str(self.nb_path), as_version=4)
        except Exception as e:
            ttk.Label(self.body, text=f"Erreur lecture .ipynb : {e}",
                       foreground="#B12727").pack(anchor="w", padx=8, pady=8)
            return

        for w in self.body.winfo_children():
            w.destroy()
        self._cell_widgets.clear()
        self._img_refs.clear()

        for idx, cell in enumerate(nb.cells):
            ctype = cell.cell_type
            src = "".join(cell.get("source", ""))
            if ctype == "markdown":
                self._render_markdown(idx, src)
            elif ctype == "code":
                self._render_code(idx, src)
            # raw cells: ignored

        self.log_fn(f"📓 {self.nb_path.name} chargé ({len(nb.cells)} cellules).")

    def _reload(self) -> None:
        self._load_notebook()

    def _reset_kernel(self) -> None:
        InteractiveShell.clear_instance()
        self.shell = InteractiveShell.instance()
        self.shell.user_ns["__file__"] = str(self.nb_path)
        for cw in self._cell_widgets:
            self._set_output(cw, "", clear=True)
        self.log_fn("🧹 Kernel réinitialisé (espace de noms vide).")

    # ------------------------------------------------------------------ cells
    def _render_markdown(self, idx: int, src: str) -> None:
        frame = ttk.Frame(self.body, padding=(8, 4))
        frame.pack(fill="x", anchor="w")
        # Très basique : titres ###, gras **, code `…` → formatage
        for raw in src.splitlines():
            line = raw.rstrip()
            if not line.strip():
                ttk.Label(frame, text="").pack(anchor="w")
                continue
            if line.startswith("# "):
                ttk.Label(frame, text=line[2:],
                           font=("Segoe UI", 14, "bold"),
                           foreground="#1F3A93").pack(anchor="w")
            elif line.startswith("## "):
                ttk.Label(frame, text=line[3:],
                           font=("Segoe UI", 12, "bold"),
                           foreground="#26577C").pack(anchor="w")
            elif line.startswith("### "):
                ttk.Label(frame, text=line[4:],
                           font=("Segoe UI", 11, "bold")).pack(anchor="w")
            elif line.startswith("- ") or line.startswith("* "):
                ttk.Label(frame, text="  • " + line[2:],
                           wraplength=1000, justify="left").pack(anchor="w")
            else:
                # Strip simple markdown emphasis for display
                clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
                clean = re.sub(r"`(.+?)`", r"\1", clean)
                ttk.Label(frame, text=clean, wraplength=1000,
                           justify="left").pack(anchor="w")

    def _render_code(self, idx: int, src: str) -> None:
        outer = ttk.Frame(self.body, padding=(8, 4))
        outer.pack(fill="x", anchor="w")

        head = ttk.Frame(outer); head.pack(fill="x")
        ttk.Label(head, text=f"In [{idx}]", foreground="#1F3A93",
                  font=("Consolas", 9, "bold")).pack(side="left")
        run_btn = ttk.Button(head, text="▶ Run")
        run_btn.pack(side="right")

        code_text = tk.Text(outer, height=max(2, min(20, src.count("\n") + 1)),
                             wrap="none", font=("Consolas", 10),
                             background="#F4F4FF", relief="solid",
                             borderwidth=1)
        code_text.insert("1.0", src)
        code_text.pack(fill="x", pady=2)

        out_frame = ttk.Frame(outer, padding=(0, 2))
        out_frame.pack(fill="x")

        cw = {"text": code_text, "out": out_frame, "idx": idx}
        run_btn.configure(command=lambda c=cw: self._run_cell(c))
        self._cell_widgets.append(cw)

    # ------------------------------------------------------------------ exec
    def _run_cell(self, cw: dict) -> None:
        src = cw["text"].get("1.0", "end-1c")
        self._set_output(cw, "", clear=True)
        # Capture stdout/stderr via shell.run_cell which redirects them
        from contextlib import redirect_stdout, redirect_stderr
        buf_out, buf_err = io.StringIO(), io.StringIO()
        # Set cwd so relative paths inside the notebook work. The notebook
        # directory has already been inserted into sys.path in __init__ so
        # that `from REST_modules import *` resolves like in Jupyter.
        old_cwd = os.getcwd()
        os.chdir(str(self.nb_path.parent.resolve()))
        try:
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                result = self.shell.run_cell(src, store_history=True)
        except Exception as e:
            self._append_output(cw, f"[exception] {e}\n", color="#B12727")
            return
        finally:
            os.chdir(old_cwd)

        if buf_out.getvalue():
            self._append_output(cw, buf_out.getvalue())
        if buf_err.getvalue():
            self._append_output(cw, buf_err.getvalue(), color="#B12727")

        # Result.result is the last expression's value
        res = getattr(result, "result", None)
        if res is not None:
            self._render_pyobj(cw, res)

        if not result.success:
            err = result.error_in_exec or result.error_before_exec
            if err:
                self._append_output(cw, f"{type(err).__name__}: {err}\n",
                                    color="#B12727")

    def run_all(self) -> None:
        for cw in self._cell_widgets:
            self._run_cell(cw)

    # ------------------------------------------------------------------ output rendering
    def _set_output(self, cw: dict, text: str, clear: bool = False) -> None:
        if clear:
            for w in cw["out"].winfo_children():
                w.destroy()
        if text:
            self._append_output(cw, text)

    def _append_output(self, cw: dict, text: str, color: str = "#222") -> None:
        if not text.strip():
            return
        lbl = tk.Text(cw["out"], height=min(20, text.count("\n") + 1),
                       wrap="word", font=("Consolas", 9),
                       background="#FFFFFF", foreground=color,
                       relief="flat", borderwidth=0)
        lbl.insert("1.0", text)
        lbl.configure(state="disabled")
        lbl.pack(fill="x", anchor="w")

    def _render_pyobj(self, cw: dict, obj) -> None:
        """Affiche un objet retourné par une cellule.

        Détecte :
          - dict avec _repr_png_ / _repr_html_ (rich output)
          - ipywidgets (placeholder)
          - sinon repr() texte
        """
        try:
            modname = type(obj).__module__ or ""
        except Exception:
            modname = ""

        if "ipywidgets" in modname or "traitlets" in modname:
            self._append_output(
                cw,
                f"[widget interactif {type(obj).__name__} — "
                f"non rendu en mode in-process]\n",
                color="#8A6D00",
            )
            return

        # Try _repr_png_ for matplotlib figures and the like
        if hasattr(obj, "_repr_png_"):
            try:
                png = obj._repr_png_()
                if isinstance(png, str):
                    png = base64.b64decode(png)
                self._render_png(cw, png)
                return
            except Exception:
                pass

        if hasattr(obj, "_repr_html_"):
            try:
                html = obj._repr_html_()
                self._append_output(cw, self._strip_html(html))
                return
            except Exception:
                pass

        # Fallback repr
        try:
            text = repr(obj)
        except Exception as e:
            text = f"<repr error: {e}>"
        if len(text) > 4000:
            text = text[:4000] + "\n…[tronqué]"
        self._append_output(cw, text + "\n")

    def _render_png(self, cw: dict, png_bytes: bytes) -> None:
        try:
            from PIL import Image, ImageTk
            img = Image.open(io.BytesIO(png_bytes))
            # Cap width to ~900 px
            if img.width > 900:
                ratio = 900 / img.width
                img = img.resize((900, int(img.height * ratio)))
            ph = ImageTk.PhotoImage(img)
            lbl = tk.Label(cw["out"], image=ph, background="#FFFFFF")
            lbl.image = ph  # keep ref
            self._img_refs.append(ph)
            lbl.pack(anchor="w", pady=4)
        except Exception as e:
            self._append_output(cw, f"[image non rendue : {e}]\n",
                                 color="#B12727")

    @staticmethod
    def _strip_html(html: str) -> str:
        text = re.sub(r"<br\s*/?>", "\n", html)
        text = re.sub(r"</p>", "\n\n", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        return text.strip() + "\n"


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1100x780")
    NotebookViewer(
        root,
        Path(__file__).resolve().parent.parent /
        "src" / "duraxell" / "REST_interface" / "REST.ipynb",
        log_fn=print,
    )
    root.mainloop()
