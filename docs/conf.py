from pathlib import Path
import re
project = 'UniVI'
author = 'Andrew J. Ashford and contributors'
copyright = '2026, UniVI contributors'
release = re.search(r'^version\s*=\s*"([^"]+)"',
                    (Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text(encoding="utf-8"),
                    re.MULTILINE).group(1)
version = '.'.join(release.split('.')[:2])
extensions = ['myst_parser', 'sphinx_copybutton', 'sphinx.ext.mathjax']
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
master_doc = 'index'
exclude_patterns = ['_build', '_fragments', 'examples', '_downloads', 'requirements.txt', 'README.md']
myst_enable_extensions = ['colon_fence', 'deflist', 'dollarmath']
myst_heading_anchors = 3
html_theme = 'sphinx_rtd_theme'
html_theme_options = {'navigation_depth': 3, 'collapse_navigation': False, 'sticky_navigation': True, 'style_nav_header_background': '#12384b'}
html_static_path = ['_static']
html_css_files = ['custom.css']
html_title = 'UniVI: multimodal single-cell analysis'
html_show_sourcelink = True
html_context = {'display_github': True, 'github_user': 'Ashford-A', 'github_repo': 'UniVI', 'github_version': 'main', 'conf_py_path': '/docs/'}
copybutton_prompt_text = r'>>> |\.\.\. |\$ '
copybutton_prompt_is_regexp = True
nitpicky = True
