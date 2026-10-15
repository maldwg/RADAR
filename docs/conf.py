"""Build RADAR's native reStructuredText documentation locally and on Read the Docs."""
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

project = "RADAR"
author = "RADAR contributors"
release = metadata["version"]
version = release
language = "en"
extensions = ["sphinx.ext.autodoc"]
source_suffix = {".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
nitpicky = True
autodoc_typehints = "none"

html_theme = "sphinx_rtd_theme"
html_title = f"RADAR {release}"
html_short_title = "RADAR"
html_logo = "../assets/branding/radar-mark.svg"
html_favicon = "../assets/branding/radar-favicon.ico"
html_static_path = ["_static"]
html_css_files = ["radar.css"]
html_theme_options = {"collapse_navigation": False, "navigation_depth": 3}
html_context = {
    "display_github": True,
    "github_user": "maldwg",
    "github_repo": "RADAR",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
html_show_sourcelink = True
html_show_copyright = False
html_show_sphinx = False
