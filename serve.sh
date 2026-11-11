#!/bin/bash
set -e
python3 -m venv ~/virtual-envs/hf-mkdocs
source ~/virtual-envs/hf-mkdocs/bin/activate
pip install -r requirements.txt
mkdocs serve
