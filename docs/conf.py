"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
import osd_masker

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "osd-masker"
copyright = "2026, Kitware, Inc."  # noqa: A001
author = "Kitware, Inc."
release = osd_masker.__version__
version = ".".join(release.split(".")[:2])

# -- Version switcher configuration ------------------------------------------
# Determine version for switcher matching based on ReadTheDocs or git branch
rtd_version = os.environ.get("READTHEDOCS_VERSION")

if rtd_version:
    # Building on ReadTheDocs - use their version identifier
    # "latest" -> main branch builds
    # "stable" -> latest release
    # "v0.26.0" -> specific tagged version
    switcher_version = rtd_version
    if rtd_version.startswith("v"):
        # Remove 'v' prefix for version matching (e.g., "v0.26.0" -> "0.26.0")
        switcher_version = rtd_version[1:]
else:
    try:
        git_path = shutil.which("git")
        if not git_path:
            raise FileNotFoundError("git executable not found in PATH")
        result = subprocess.run(  # noqa: S603
            [git_path, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        branch = result.stdout.strip()

        # Map branches to switcher versions
        # main branch -> "latest", other branches -> package release version
        switcher_version = "latest" if branch == "main" else release
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Git not available or not in a repo - use release version
        switcher_version = release

# -- Contextual date label for landing page ----------------------------------
# Provide a contextual label for the build date shown on the landing page.
# Sphinx's |today| already shows the build date for each specific version.
if rtd_version == "latest":
    date_label = "Latest Build"
elif rtd_version == "stable":
    date_label = "Stable Release"
elif rtd_version:
    date_label = f"{rtd_version} Docs Built"
else:
    date_label = "Date"

rst_prolog = f"""
.. |date_label| replace:: **{date_label}**
"""

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

today_fmt = "%B %d, %Y"  # e.g., "February 11, 2026"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.imgconverter",
    "sphinx-prompt",
    "sphinx_copybutton",
    "sphinx_click",
]
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True

suppress_warnings = [
    # Suppressing duplicate label warning from autosectionlabel extension.
    # This happens a lot across files that happen to talk about the same
    # topics.
    "autosectionlabel.*",
]

# Autosummary templates reference link:
# https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion/tree/master
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Pending release note fragments are collected into a versioned page at
    # release time; they should not be processed individually by Sphinx.
    "release_notes/pending_release/*",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []

# -- LaTeX engine ------------------------------------------------------------
latex_engine = "lualatex"
latex_elements = {
    # Use fontspec with default Latin Modern fonts (bundled with texlive)
    # instead of Sphinx's default FreeSerif/FreeSans/FreeMono which may
    # not be found by luaotfload's font resolver.
    "fontpkg": r"\usepackage{fontspec}",
}
# Use makeindex instead of xindy (Sphinx's default for lualatex) since
# xindy may not be available in all CI environments.
latex_use_xindy = False
