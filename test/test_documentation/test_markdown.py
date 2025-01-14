import json
import re
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Tuple
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest


@pytest.mark.allowed_to_fail
def test_markdown():
    n_serious = validate_files(list(enumerate_files()), return_type='summary', output='warn').get(1, 0)
    assert n_serious == 0


# taken from https://stackoverflow.com/a/3809435
_URL_REGEX = re.compile(
    r'https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_+.~#?&/=]*'
)

_HEADER_REGEX = re.compile(r'#+ +(.+?)( +#* *)?')

_CATABRA_GH_REGEX = re.compile(r'https?://(?:www\.)?github\.com/risc-mi/catabra(.*)')

_CATABRA_RTD_REGEX = re.compile(r'https?://catabra\.readthedocs\.io/en/latest(.*)')

_CODE_REGEX = re.compile(r'.*\.(?:py|csv)(?:#.*)?')

_DOC_REGEX = re.compile(r'(?:/?|.*github\.com/risc-mi/catabra/tree/main/)doc/.+\.md')

_ROOT = Path(__file__).parent
while _ROOT.stem != 'catabra':
    _ROOT = _ROOT.parent
    if _ROOT == Path.root:
        raise RuntimeError('No CaTabRa root directory found.')

_SPHINX_DOCS = _ROOT / 'doc/sphinx-docs'
if not _SPHINX_DOCS.exists():
    warnings.warn(f'sphinx-docs directory {_SPHINX_DOCS.as_posix()} does not exist. Is the path correct?')

ISSUES = {
    0: ('section reference embedded as image', 1),
    1: ('relative non-image link', 1),
    2: ('bare hyperlink in markdown cell', 0),
    3: ('section header not found', 1),
    4: ('image link points to non-image resource', 1),
    5: ('image reference not embedded as image', 0),
    6: ('resource not found', 1),
    7: ('resource not found locally', 1),
    8: ('resource only found locally', 0),
    9: ('source code reference not enclosed in ``', 0),
    10: ('code block without closing ```', 1),
    11: ('link to github instead of readthedocs', 0),
    12: ('readthedocs document not found', 0),
}


def parse_cell(path: Path, text: Iterable[str], cell_id=None, markdown: bool = True) -> Tuple[list, list, list]:
    abs_refs = []
    sec_refs = []
    msgs = []

    root = 'https://github.com/risc-mi/catabra/tree/main/' + path.parent.relative_to(_ROOT).as_posix() + '/'

    if isinstance(text, str):
        text = text.split('\n')
    code_block = -1
    for i, ln in enumerate(text, start=1):
        ln = ln.strip('\n')
        url_starts = set()
        if markdown:
            if ln.startswith('```'):
                if code_block >= 0:
                    if len(ln) == 3:
                        code_block = -1
                elif len(ln) == 3 or not ln.endswith('```'):
                    code_block = i

            if code_block < 0:
                for caption, url, is_img, span, url_start in _iterate_inline_links(ln):
                    url_starts.add(url_start)
                    pos = (cell_id, i, span)
                    if is_img:
                        if not (url[-4:].lower() in ('.png', '.jpg', '.gif', '.bmp')
                                or url.startswith('https://img.shields.io/')):
                            msgs.append((pos, 4, url))
                    elif url[-4:].lower() in ('.png', '.jpg', '.gif', '.bmp'):
                        msgs.append((pos, 5, url))
                    if url.startswith('#'):
                        if is_img:
                            msgs.append((pos, 0, url))
                        sec_refs.append((pos, url[1:]))
                    elif url.startswith('http://') or url.startswith('https://'):
                        abs_refs.append((pos, caption, url))
                    else:
                        if not is_img:
                            msgs.append((pos, 1, url))
                        abs_refs.append((pos, caption, root + url))

        for url, is_img, start, stop in _iterate_bare_links(ln):
            if start not in url_starts:
                pos = (cell_id, i, (start, stop))
                if markdown and code_block < 0 and not is_img:
                    msgs.append((pos, 2, url))
                abs_refs.append((pos, '', url))

    if code_block >= 0:
        msgs.append(((cell_id, code_block, (0, 3)), 10, ''))

    return abs_refs, sec_refs, msgs


def validate_file(path: Path) -> list:
    abs_refs = []
    sec_refs = []
    headers = []
    msgs = []

    if path.suffix.lower() == '.ipynb':
        with open(path, mode='rt') as f:
            cells = json.load(f).get('cells', [])
        for i, cell in enumerate(cells):
            md = cell.get('cell_type') == 'markdown'
            src = cell.get('source', [])
            if md and len(src) == 1:
                match = _HEADER_REGEX.fullmatch(src[0])
                if match is not None:
                    headers.append(match.groups()[0].replace(' ', '-'))
                    continue
            a, s, m = parse_cell(path, src, cell_id=(i, cell.get('id')), markdown=md)
            abs_refs.extend(a)
            sec_refs.extend(s)
            msgs.extend(m)
    else:
        with open(path, mode='rt') as f:
            text = f.readlines()
        abs_refs, sec_refs, msgs = parse_cell(path, text, markdown=(path.suffix.lower() == '.md'))

    for pos, url in sec_refs:
        if url not in headers:
            msgs.append((pos, 3, url))

    url_status = {url: status for url, status in check_urls({url for _, _, url in abs_refs})}
    for pos, caption, url in abs_refs:
        exists, exists_local = url_status[url]
        if exists:
            if exists_local is False:
                msgs.append((pos, 7, url))
        else:
            if _CATABRA_RTD_REGEX.fullmatch(url):
                msgs.append((pos, 12, url))
            elif exists_local is True:
                msgs.append((pos, 8, url))
            else:
                msgs.append((pos, 6, url))
        if not (_CODE_REGEX.fullmatch(url) is None or caption == ''
                or (caption.startswith('`') and caption.endswith('`'))):
            msgs.append((pos, 9, caption))
        if (url.endswith('.md') or url.endswith('.ipynb')) and _CATABRA_GH_REGEX.fullmatch(url):
            msgs.append((pos, 11, url))

    msgs.sort(key=(lambda x: (-1 if x[0][0] is None else x[0][0][0], x[0][1], x[0][2][0])))

    return msgs


def validate_files(files, return_type: str = 'none', output: str = 'print') -> Optional[dict]:
    if isinstance(files, (str, Path)):
        files = [files]
    if not isinstance(files, (list, set)):
        files = list(files)

    n = len(files)
    summary = {}
    messages = {}
    for i, f in enumerate(files, start=1):
        if output == 'print':
            print(f'### {f.as_posix()} ({i}/{n})')
        for (cell_id, line, (start, end)), idx, text in validate_file(f):
            category, level = ISSUES[idx]
            summary[level] = summary.get(level, 0) + 1
            messages.setdefault(f.as_posix(), []).append(((cell_id, line, (start, end)), idx, text))

            if output == 'print':
                print('   ', _repr_msg(cell_id, line, start, end, idx, text))
            elif output == 'warn':
                if isinstance(cell_id, tuple):
                    cell_id = cell_id[-1]
                if cell_id is None:
                    file = f.as_posix()
                else:
                    file = f.as_posix() + ':' + str(cell_id)
                text = f'{category} ({idx}; {start}-{end}): {text}'
                warnings.warn_explicit(text, Exception if level > 0 else Warning, file, line)
        if output == 'print':
            print()

    if output == 'print':
        print(f'Validated {n} files: {summary.get(0, 0)} warnings, {summary.get(1, 0)} serious issues')
    if return_type == 'summary':
        return summary
    elif return_type == 'messages':
        return messages


def enumerate_files(d: Path = _ROOT, file_ext=('.py', '.md', '.ipynb', '.txt'), tracked_only: bool = True,
                    include_sphinx_docs: bool = False):
    if tracked_only:
        res = subprocess.run(['git', 'ls-files'], cwd=_ROOT, capture_output=True, text=True)
        if res.returncode == 0:
            d = d.absolute()
            for f0 in res.stdout.split('\n'):
                if f0:
                    f = _ROOT / f0      # all paths are relative to `_ROOT`, because `cwd=_ROOT`
                    if f.suffix.lower() in file_ext and f.is_relative_to(d) \
                            and (include_sphinx_docs or not f.is_relative_to(_SPHINX_DOCS)):
                        yield f
            return
        else:
            raise ValueError(res.stderr)

    for f in d.iterdir():
        if include_sphinx_docs or not f.is_relative_to(_SPHINX_DOCS):
            if f.is_dir():
                yield from enumerate_files(d=f, file_ext=file_ext, tracked_only=tracked_only)
            elif f.suffix.lower() in file_ext:
                yield f


def check_urls(urls):
    if urls:
        with ThreadPoolExecutor(min(1000, len(urls))) as executor:
            futures = {executor.submit(_check_url, url): url for url in urls}
            for future in as_completed(futures):
                yield futures[future], future.result()


def _iterate_inline_links(text: str):
    start = 1
    last_link_end = -1
    while start + 2 < len(text):
        i = text.find('](', start)
        if i < 0:
            return

        # find matching "["
        brackets = 1
        last_bracket = None
        for j in range(i - 1, last_link_end, -1):
            if text[j] == '[':
                brackets -= 1
                last_bracket = j
            elif text[j] == ']':
                brackets += 1
            if brackets == 0:
                break
        if last_bracket is None:
            start = i + 1
            continue
        else:
            caption = text[last_bracket + 1:i]
            is_img = last_bracket > last_link_end and text[last_bracket - 1] == '!'
            span = last_bracket

        # find matching ")"
        # skip over whitespace
        for i in range(i + 2, len(text)):
            if text[i] != ' ':
                break
        brackets = 1
        last_bracket = None
        whitespace = None
        for j in range(i, len(text)):
            if whitespace is None:
                if text[j] == ' ':
                    whitespace = j
                elif text[j] == '(':
                    brackets += 1
                elif text[j] == ')':
                    brackets -= 1
                    last_bracket = j
                if brackets == 0:
                    break
            else:
                if text[j] == ')':
                    last_bracket = j
                    break
                elif text[j] != ' ':
                    break

        if last_bracket is None:
            start = i + 1
            continue
        else:
            url = text[i:(last_bracket if whitespace is None else whitespace)]
            span = (span, last_bracket)
            last_link_end = last_bracket
            start = last_link_end + 2
            yield caption, url, is_img, span, i


def _iterate_bare_links(text: str):
    for match in _URL_REGEX.finditer(text):
        start, stop = match.span()
        url = match.group()
        is_img = False
        if 0 < start and stop < len(text) and text[start - 1] == '"' and text[stop] == '"':
            is_img = 10 <= start and text[start - 10:start] == '<img src="'
        elif not (0 < start and stop < len(text) and text[start - 1] == '<' and text[stop] == '>'):
            while url:
                if url[-1] in '(.':
                    url = url[:-1]
                    stop -= 1
                elif url[-1] == ')':
                    brackets = -1
                    for i in range(len(url) - 2, -1, -1):
                        if url[i] == ')':
                            brackets -= 1
                        elif url[i] == '(':
                            brackets += 1
                            if brackets == 0:
                                break
                    if brackets == 0:
                        break
                    else:
                        url = url[:-1]
                        stop -= 1
                else:
                    break
        yield url, is_img, start, stop


def _check_url(url: str) -> Tuple[bool, Optional[bool]]:
    try:
        with urlopen(url) as connection:
            exists = connection.status == 200
    except HTTPError as ex:
        exists = ex.code == 403
    except:     # noqa
        exists = False

    match = _CATABRA_GH_REGEX.fullmatch(url)
    if match is not None:
        match = match.groups()[0]
        if match.startswith('/'):
            match = match[1:]
            if match:
                if match.startswith('tree/main') or match.startswith('blob/main'):
                    match = match[9:]
                    if match.startswith('/'):
                        match = match[1:]
                    match = (_ROOT / match).exists()
                else:
                    match = match == '.git'
            else:
                # CaTabRa root
                match = True
        else:
            # CaTabRa root or other repository (catabra-pandas, ...)
            match = True

    return exists, match


def _repr_msg(cell_id, line, start, end, idx, text) -> str:
    if isinstance(cell_id, tuple):
        cell_id = cell_id[-1]
    if cell_id is None:
        cell_id = ''
    else:
        cell_id = str(cell_id) + ':'
    category, level = ISSUES[idx]
    return f'{"!" if level > 0 else " "} {cell_id}{line}:{start}-{end} {category} ({idx}): {text}'
