"""
Fix gold_standard_dev_24.json — strips the non-JSON text appended after the valid JSON array.
Writes the fixed file in-place and saves a .bak backup.
"""
import json
import pathlib
import shutil
import re

path = pathlib.Path("gold_standard_dev_24.json")
raw = path.read_text(encoding="utf-8", errors="replace")

# Find the last ']' that closes the top-level array
# Everything after it is draft text appended to the file
last_bracket = raw.rfind("]")
valid_json = raw[: last_bracket + 1].rstrip()

# Verify it parses cleanly
data = json.loads(valid_json)
print(f"Valid items: {len(data)}")
print(f"First: {data[0].get('question', '')[:70]}")
print(f"Last:  {data[-1].get('question', '')[:70]}")

# Backup original
shutil.copy(path, path.with_suffix(".json.bak"))
print(f"Backup written: {path.with_suffix('.json.bak')}")

# Write fixed file
path.write_text(valid_json, encoding="utf-8")
print(f"Fixed file written: {path}")
