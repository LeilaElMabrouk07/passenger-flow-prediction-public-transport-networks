import re
from pathlib import Path

FILENAME_RE = re.compile(r"NBT(?P<yy>\d{2})(?P<code>[A-Za-z]+)", re.IGNORECASE)

def parse_year_and_daycode(xlsx_path: str):
    name = Path(xlsx_path).stem
    m = FILENAME_RE.search(name)
    if not m:
        return None, None
    yy = int(m.group("yy"))
    year = 2000 + yy
    day_code = m.group("code").upper()
    return year, day_code

