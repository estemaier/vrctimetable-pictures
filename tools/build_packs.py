#!/usr/bin/env python3
"""VRCTimetable picture robot (Stage 11B, 2026-10-05).

Turns the picture list the VRCTimetable sheet publishes (pictures.json) into "picture packs":
text files a VRCTimetable board downloads with VRChat's text downloader (a separate queue from
picture downloads, so picture prefabs in the same world are not slowed down) and turns back into
GPU-compressed textures (DXT1 for posters, DXT5 for logos: about 6x less video memory than plain
pictures).

Runs in GitHub Actions in the organiser's own GitHub account (workflow vrctimetable-pictures.yml),
which pushes the output folder to the repository's branch "packs" (one commit, replaced each run);
the VRCTimetable sheet then copies new or changed files into the schedule's own gist, where boards
read them — the same address as the schedule, so pictures reach everyone the schedule reaches
(GitHub Pages is blocked by some internet providers). Locally for tests:
    python build_packs.py pictures.json out_folder [previous_folder]
  previous_folder: the last run's output (the "packs" branch): pictures made then are kept instead of
  downloaded again (also when their link has stopped working — links copied from Discord expire),
  unless the list asks for a fresh run ("fresh" differs) or the sizes changed; and every picture
  keeps the pack it was in, so a change rewrites only the packs it touches (the gist keeps every
  version of its files, so it grows only by those).

pictures.json:
    {"v": 1, "version": "<the sheet's hash of this list>", "fresh": "<stamp of the last 'make again'>",
     "pictures": [{"id": "p0123456789", "kind": "poster" | "logo", "url": "https://...", "order": 29849040}, ...]}
  order = the event's start (Unix minutes) for posters: new posters are packed in time order.

Output, one folder:
    index.json                    which pack holds which picture (the board reads it first, and again
                                  when the schedule names pictures it does not know; the sheet reads
                                  it to report broken pictures and to see which packs changed)
    pack-00.txt ... pack-15.txt   the packs; on a first run the logos come first (pack-00), then the
                                  posters by time; later runs keep everything where it was

index.json: {"v": 1, "version": <pictures.json version>, "fresh": <its stamp>, "shape": <sizes used>,
             "packs": count, "where": {id: pack}, "sizes": {id: [w, h]}, "versions": [hash of each pack],
             "status": {id: "ok" | "<code>|<details>"}}
  codes: download (could not be downloaded), notpicture, toolarge (over 40 megapixels),
  filesize (over 25 MB), noroom (more pictures than the packs hold) — the sheet words them.

Pack format "VTP1" (all ASCII):
    VTP1\n<length of the JSON header>\n<JSON header><base64 block><base64 block>...
  header: {"v": 1, "pack": n, "pictures": [{"id", "kind", "w", "h", "fmt", "mips", "at", "len"}]}
  at / len: where the picture's base64 block sits in the text after the header. Each block is the
  texture's raw data in Unity's layout (rows bottom-up, every mip level after the other), so the
  board needs only Substring, Convert.FromBase64String and Texture2D.LoadRawTextureData.
  fmt: 1 = DXT1 (no alpha), 5 = DXT5 (alpha). Width and height are multiples of 4.
"""
import base64
import hashlib
import io
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageOps

FORMAT = "VTP1"
POSTER_MAX = 1024           # longest side of a poster, px
LOGO_SIZE = 256             # logos: fitted into this square (they show at about 48 px; sharp when a visitor looks closer)
PACK_BUDGET = 2_000_000     # raw picture bytes per pack (the text is 4/3 of it)
MAX_PACKS = 16              # the board knows this many pack links (VRCUrls fixed at upload)
MAX_DOWNLOAD = 25 * 1024 * 1024
MAX_SOURCE_PIXELS = 40_000_000
USER_AGENT = "VRCTimetable-picture-robot/1.0 (+https://tektinok-ltd.booth.pm)"
DXT1, DXT5 = 1, 5


# ---------------------------------------------------------------- downloading

def direct_link(url):
    """Share links of Google Drive and Dropbox point at a web page; turn them into the file itself."""
    u = url.strip()
    m = re.match(r"https?://drive\.google\.com/file/d/([^/?#]+)", u) or re.match(r"https?://drive\.google\.com/open\?id=([^&#]+)", u)
    if m:
        return "https://drive.google.com/uc?export=download&id=" + m.group(1)
    if re.match(r"https?://(www\.)?dropbox\.com/", u):
        parts = urllib.parse.urlsplit(u)
        q = urllib.parse.parse_qs(parts.query)
        q["dl"] = ["1"]
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(q, doseq=True), ""))
    return u


def fetch(url):
    """The picture's bytes; a local path or file: URL works too (tests)."""
    if url.startswith("file:"):
        url = urllib.request.url2pathname(urllib.parse.urlsplit(url).path)
    if os.path.exists(url):
        with open(url, "rb") as f:
            return f.read(MAX_DOWNLOAD + 1)
    req = urllib.request.Request(direct_link(url), headers={"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.5"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read(MAX_DOWNLOAD + 1)
    return data


def open_picture(data):
    if len(data) > MAX_DOWNLOAD:
        raise ValueError("filesize|over 25 MB")
    Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS
    try:
        img = Image.open(io.BytesIO(data))
        img.seek(0)                      # GIF / WebP: the first frame
        img = ImageOps.exif_transpose(img)
        img.load()
    except Image.DecompressionBombError:
        raise ValueError("toolarge|over 40 megapixels")
    except Exception:
        raise ValueError("notpicture|JPG, PNG, WebP or GIF expected")
    return img


def problem(e):
    """A picture's status when it failed: "<code>|<details>" (the sheet shows the code in its own language)."""
    if isinstance(e, ValueError) and "|" in str(e):
        return str(e)
    if isinstance(e, urllib.error.HTTPError):
        return "download|HTTP %d" % e.code
    return "download|%s" % e.__class__.__name__


# ---------------------------------------------------------------- shaping

def multiple_of_4(n):
    return max(4, (n // 4) * 4)


def shape_poster(img):
    """RGB, longest side at most POSTER_MAX, both sides multiples of 4 (never enlarged beyond the source)."""
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, POSTER_MAX / max(w, h))
    tw, th = multiple_of_4(round(w * scale)), multiple_of_4(round(h * scale))
    if (tw, th) != (w, h):
        img = img.resize((tw, th), Image.Resampling.LANCZOS)
    return img


def shape_logo(img):
    """RGBA, fitted into a LOGO_SIZE square on a transparent background (no stretching)."""
    img = img.convert("RGBA")
    w, h = img.size
    scale = LOGO_SIZE / max(w, h)
    tw, th = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((tw, th), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (LOGO_SIZE, LOGO_SIZE), (0, 0, 0, 0))
    canvas.paste(img, ((LOGO_SIZE - tw) // 2, (LOGO_SIZE - th) // 2))
    a = np.asarray(canvas).copy()
    # Transparent pixels take the logo's average colour: no dark fringes when the board filters it.
    opaque = a[..., 3] > 8
    if opaque.any():
        mean = a[opaque][:, :3].mean(axis=0).round().astype(np.uint8)
        a[~opaque, :3] = mean
    return Image.fromarray(a, "RGBA")


def mip_sizes(w, h):
    """Unity's mip chain for a w x h texture: floor(log2(max)) + 1 levels, each side halved down to 1."""
    count = int(math.floor(math.log2(max(w, h)))) + 1
    return [(max(1, w >> i), max(1, h >> i)) for i in range(count)]


# ---------------------------------------------------------------- DXT encoding (numpy, deterministic)

def to_blocks(a):
    """H x W x C (multiples of 4) -> (blocks, 16, C), blocks row by row, pixels row by row inside a block."""
    hgt, wid, ch = a.shape
    return a.reshape(hgt // 4, 4, wid // 4, 4, ch).transpose(0, 2, 1, 3, 4).reshape(-1, 16, ch)


def pad4(a):
    hgt, wid = a.shape[:2]
    ph, pw = (-hgt) % 4, (-wid) % 4
    if ph or pw:
        a = np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return a


def expand565(c):
    r = (c >> 11) & 31
    g = (c >> 5) & 63
    b = c & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], axis=-1).astype(np.float32)


def encode_color(px, weight):
    """px: (n, 16, 3) float32, weight: (n, 16) (0 = ignore the pixel). Returns (c0, c1, indices) as uint arrays."""
    n = px.shape[0]
    wsum = weight.sum(axis=1, keepdims=True)
    wsum_safe = np.where(wsum > 0, wsum, 1.0)
    w3 = weight[..., None]
    mean = (px * w3).sum(axis=1) / wsum_safe                           # (n, 3)
    mean = np.where(wsum > 0, mean, px.mean(axis=1))
    d = px - mean[:, None, :]
    cov = np.einsum("nki,nkj->nij", d * w3, d)                         # (n, 3, 3)
    v = np.tile(np.array([0.577, 0.577, 0.577], np.float32), (n, 1))
    for _ in range(8):                                                  # principal axis by power iteration
        v = np.einsum("nij,nj->ni", cov, v)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.where(norm > 1e-6, v / np.maximum(norm, 1e-6), np.float32(0.577))
    proj = np.einsum("nki,ni->nk", d, v)
    big = np.float32(1e9)
    pmin = np.where(weight > 0, proj, big).min(axis=1)
    pmax = np.where(weight > 0, proj, -big).max(axis=1)
    pmin = np.where(pmin > big / 2, 0, pmin)
    pmax = np.where(pmax < -big / 2, 0, pmax)
    hi = np.clip(mean + pmax[:, None] * v, 0, 255)
    lo = np.clip(mean + pmin[:, None] * v, 0, 255)

    def q565(c):
        r = np.clip(np.round(c[:, 0] * 31 / 255), 0, 31).astype(np.uint32)
        g = np.clip(np.round(c[:, 1] * 63 / 255), 0, 63).astype(np.uint32)
        b = np.clip(np.round(c[:, 2] * 31 / 255), 0, 31).astype(np.uint32)
        return (r << 11) | (g << 5) | b

    c0 = q565(hi)
    c1 = q565(lo)
    swap = c0 < c1
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
    p0, p1 = expand565(c0), expand565(c1)
    pal = np.stack([p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3], axis=1)   # (n, 4, 3) four-colour mode
    dist = ((px[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(axis=-1)      # (n, 16, 4)
    idx = dist.argmin(axis=-1).astype(np.uint32)
    idx = np.where((c0 == c1)[:, None], 0, idx)                              # one colour: index 0 everywhere
    return c0, c1, idx


def pack_indices(idx, bits):
    shifts = (np.arange(16, dtype=np.uint64) * bits)
    return (idx.astype(np.uint64) << shifts).sum(axis=1)


def encode_dxt1(a):
    """a: H x W x 3 uint8 (already flipped to Unity's bottom-up rows) -> DXT1 bytes."""
    blk = to_blocks(pad4(a)).astype(np.float32)
    c0, c1, idx = encode_color(blk, np.ones(blk.shape[:2], np.float32))
    out = np.zeros((blk.shape[0], 8), np.uint8)
    out[:, 0] = c0 & 255
    out[:, 1] = c0 >> 8
    out[:, 2] = c1 & 255
    out[:, 3] = c1 >> 8
    bits = pack_indices(idx, 2)
    for k in range(4):
        out[:, 4 + k] = (bits >> np.uint64(8 * k)) & np.uint64(255)
    return out.tobytes()


def encode_dxt5(a):
    """a: H x W x 4 uint8 (bottom-up rows) -> DXT5 bytes (8-byte alpha block + 8-byte colour block)."""
    blk = to_blocks(pad4(a))
    alpha = blk[..., 3].astype(np.int32)
    a0 = alpha.max(axis=1)
    a1 = alpha.min(axis=1)
    # Eight-value mode (a0 > a1): a0, a1, then six steps between them.
    steps = np.stack([a0, a1] + [((7 - k) * a0 + k * a1) // 7 for k in range(1, 7)], axis=1)   # palette order: idx 0..7
    dist = np.abs(alpha[:, :, None] - steps[:, None, :])
    aidx = dist.argmin(axis=-1).astype(np.uint32)
    aidx = np.where((a0 == a1)[:, None], 0, aidx)
    abits = pack_indices(aidx, 3)
    weight = (blk[..., 3] > 8).astype(np.float32)
    c0, c1, idx = encode_color(blk[..., :3].astype(np.float32), weight)
    out = np.zeros((blk.shape[0], 16), np.uint8)
    out[:, 0] = a0
    out[:, 1] = a1
    for k in range(6):
        out[:, 2 + k] = (abits >> np.uint64(8 * k)) & np.uint64(255)
    out[:, 8] = c0 & 255
    out[:, 9] = c0 >> 8
    out[:, 10] = c1 & 255
    out[:, 11] = c1 >> 8
    bits = pack_indices(idx, 2)
    for k in range(4):
        out[:, 12 + k] = (bits >> np.uint64(8 * k)) & np.uint64(255)
    return out.tobytes()


def encode_texture(img, fmt):
    """Every mip level, Unity's layout (bottom-up rows), concatenated."""
    w, h = img.size
    data = bytearray()
    for (mw, mh) in mip_sizes(w, h):
        level = img if (mw, mh) == (w, h) else img.resize((mw, mh), Image.Resampling.LANCZOS)
        arr = np.asarray(level.transpose(Image.Transpose.FLIP_TOP_BOTTOM))
        data += encode_dxt1(arr[..., :3]) if fmt == DXT1 else encode_dxt5(arr)
    return bytes(data), len(mip_sizes(w, h))


def expected_size(w, h, fmt):
    block = 8 if fmt == DXT1 else 16
    return sum(max(1, (mw + 3) // 4) * max(1, (mh + 3) // 4) * block for (mw, mh) in mip_sizes(w, h))


# ---------------------------------------------------------------- the last run's pictures

SHAPE = "p%d-l%d-dxt" % (POSTER_MAX, LOGO_SIZE)   # pictures made with other sizes are made again


def read_pack(text):
    """A pack's header and where its blocks start, or (None, 0)."""
    if not text.startswith(FORMAT + "\n"):
        return None, 0
    nl = text.index("\n", len(FORMAT) + 1)
    hl = int(text[len(FORMAT) + 1:nl])
    return json.loads(text[nl + 1:nl + 1 + hl]), nl + 1 + hl


def load_previous(folder, fetcher=fetch, log=print):
    """
    The last run's output (a folder, or an address ending in "/"): its pictures by id →
    (kind, w, h, fmt, mips, base64 block), its "fresh" stamp and where each picture was
    (id → pack); ({}, None, {}) when there is none.
    """
    try:
        index = json.loads(fetcher(os.path.join(folder, "index.json") if os.path.isdir(folder) else folder + "index.json").decode("utf-8"))
    except Exception:
        return {}, None, {}
    where = {k: int(v) for k, v in index.get("where", {}).items()}
    if index.get("shape") != SHAPE:
        log("the last run's pictures were made with other sizes: every picture is made again")
        return {}, None, where
    status = index.get("status", {})
    kept = {}
    for n in range(int(index.get("packs", 0))):
        name = "pack-%02d.txt" % n
        try:
            text = fetcher(os.path.join(folder, name) if os.path.isdir(folder) else folder + name).decode("ascii")
            header, body = read_pack(text)
        except Exception:
            continue
        if header is None:
            continue
        for e in header.get("pictures", []):
            b64 = text[body + e["at"]: body + e["at"] + e["len"]]
            if status.get(e["id"]) == "ok" and len(b64) == e["len"]:
                kept[e["id"]] = (e["kind"], e["w"], e["h"], e["fmt"], e["mips"], b64)
    return kept, index.get("fresh"), where


def assign_packs(ready, prev_where):
    """
    Pictures keep the pack they were in (they fitted together then), so a change rewrites only the
    packs it touches; the others fill the first pack with room — logos first, then posters by time
    (on a first run: the logos in pack 0, then the posters in time order). An emptied pack in the
    middle stays (the others keep their numbers); empty ones at the end go. Within a pack the
    pictures are in id order, so its text depends only on which pictures it holds.
    Returns (packs, ids that found no room).
    """
    packs = []
    used = []

    def grow(n):
        while len(packs) <= n:
            packs.append([])
            used.append(0)

    later = []
    for r in ready:
        n = prev_where.get(r[2])
        if n is not None and 0 <= n < MAX_PACKS:
            grow(n)
            if used[n] + r[9] <= PACK_BUDGET:
                packs[n].append(r)
                used[n] += r[9]
                continue
        later.append(r)
    later.sort(key=lambda r: (r[0], r[1], r[2]))
    noroom = []
    for r in later:
        for n in range(len(packs)):
            if used[n] + r[9] <= PACK_BUDGET:
                packs[n].append(r)
                used[n] += r[9]
                break
        else:
            if len(packs) < MAX_PACKS:
                packs.append([r])
                used.append(r[9])
            else:
                noroom.append(r[2])
    while packs and not packs[-1]:
        packs.pop()
    if not packs:
        packs = [[]]
    for p in packs:
        p.sort(key=lambda r: r[2])
    return packs, noroom


# ---------------------------------------------------------------- packing

def build(pictures_json, out_dir, fetcher=fetch, log=print, previous=None):
    """previous: the last run's output (folder or address): its pictures are kept, and where they were."""
    if os.path.exists(pictures_json):
        with open(pictures_json, "r", encoding="utf-8") as f:
            spec = json.load(f)
    else:   # the repository before the sheet's first picture list: an empty output, not a failed run
        spec = {"v": 1, "version": "", "pictures": []}
    version = str(spec.get("version", ""))
    fresh = str(spec.get("fresh", "") or "")   # the sheet's "Make the pictures again": nothing is kept
    items = spec.get("pictures", [])
    kept = {}
    prev_where = {}
    if previous:
        kept, prev_fresh, prev_where = load_previous(previous, fetcher, log)
        if kept and (prev_fresh or "") != fresh:
            log("a fresh run was asked for: every picture is made again")
            kept = {}
    status = {}
    ready = []   # (kind rank, order, id, kind, w, h, fmt, mips, b64, raw bytes)
    seen = set()
    for it in items:
        pid = str(it.get("id", "")).strip()
        kind = it.get("kind", "poster")
        url = str(it.get("url", "")).strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        old = kept.get(pid)
        if old and old[0] == kind:
            _, w, h, fmt, mips, b64 = old
            ready.append((0 if kind == "logo" else 1, int(it.get("order", 0) or 0), pid, kind, w, h, fmt, mips, b64, expected_size(w, h, fmt)))
            status[pid] = "ok"
            log("kept  %-12s %-6s %4dx%-4d (made before)" % (pid, kind, w, h))
            continue
        try:
            img = open_picture(fetcher(url))
            shaped = shape_logo(img) if kind == "logo" else shape_poster(img)
            fmt = DXT5 if kind == "logo" else DXT1
            raw, mips = encode_texture(shaped, fmt)
            assert len(raw) == expected_size(shaped.size[0], shaped.size[1], fmt)
            ready.append((0 if kind == "logo" else 1, int(it.get("order", 0) or 0), pid, kind,
                          shaped.size[0], shaped.size[1], fmt, mips, base64.b64encode(raw).decode("ascii"), len(raw)))
            status[pid] = "ok"
            log("ok    %-12s %-6s %4dx%-4d %s" % (pid, kind, shaped.size[0], shaped.size[1], url[:80]))
        except Exception as e:   # one broken picture never stops the others
            msg = problem(e)
            status[pid] = msg
            log("FAIL  %-12s %-6s %s  <- %s" % (pid, kind, msg, url[:80]))

    packs, noroom = assign_packs(ready, prev_where)
    for pid in noroom:
        status[pid] = "noroom|more pictures than %d packs hold" % MAX_PACKS

    def pack_text(n, entries):
        header = {"v": 1, "pack": n, "pictures": []}
        body = []
        at = 0
        for r in entries:
            header["pictures"].append({"id": r[2], "kind": r[3], "w": r[4], "h": r[5], "fmt": r[6], "mips": r[7], "at": at, "len": len(r[8])})
            body.append(r[8])
            at += len(r[8])
        js = json.dumps(header, separators=(",", ":"), ensure_ascii=True)
        return FORMAT + "\n" + str(len(js)) + "\n" + js + "".join(body)

    texts = [pack_text(n, p) for n, p in enumerate(packs)]
    versions = [hashlib.sha1(t.encode("ascii")).hexdigest()[:12] for t in texts]
    where = {r[2]: n for n, p in enumerate(packs) for r in p}
    sizes = {r[2]: [r[4], r[5]] for p in packs for r in p}   # the board lays out a poster's place before it arrives
    index = {"v": 1, "version": version, "fresh": fresh, "shape": SHAPE, "packs": len(packs), "where": where, "sizes": sizes,
             "versions": versions, "status": status}

    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        if re.match(r"pack-\d\d\.txt$", name):
            os.remove(os.path.join(out_dir, name))
    for n, t in enumerate(texts):
        with open(os.path.join(out_dir, "pack-%02d.txt" % n), "w", encoding="ascii", newline="") as f:
            f.write(t)
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    log("%d pictures in %d packs (%s), %d problems" % (len(where), len(packs),
        ", ".join("%.1f MB" % (len(t) / 1e6) for t in texts), sum(1 for s in status.values() if s != "ok")))
    return index


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(2)
    build(sys.argv[1], sys.argv[2], previous=sys.argv[3] if len(sys.argv) == 4 else None)
