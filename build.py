#!/usr/bin/env python3
"""
Jour's weather map, built from open data.

Every run brings the map files up to date and writes them out as a static
site: the forecast for the next two and a half days from DWD's ICON-D2
(2 km, Germany and the Alps), ICON-EU (7 km, Europe) and NOAA's GFS (25 km,
everywhere), and the radar of the last two hours from MeteoSwiss, DWD and
NOAA's MRMS. All of it is free to use, also in a paid app, as long as the
sources are named.

What comes out is one small file per piece of map: a square of 128 by 128
grid points, every hour (or radar frame) of one layer, one byte a point,
each frame stored as its difference from the one before and compressed.
`manifest.json` says what there is and where. The app reads only the
pieces it is looking at.

    python build.py --state state --site site

`state` is kept between runs (the finished forecast pieces, the radar frames
already read), so a run only fetches what is new.
"""

from __future__ import annotations

import argparse
import bz2
import concurrent.futures as cf
import datetime as dt
import gzip
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

import eccodes
import h5py
import numpy as np
import pyproj

AGENT = "jour-map/1.0 (+https://github.com/hinrichb/jour-map)"
CHUNK = 128
MISSING = 255
LAYERS = ("rain", "temperature", "wind", "clouds")
# How far the forecast reaches from its run: the app wants two days from
# now, and a run is a few hours old by the time it is out.
HOURS = int(os.environ.get("JOUR_HOURS", "60"))
RADAR_FRAMES = 13            # two hours, every ten minutes
RADAR_STEP = 600


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


# MARK: - Fetching

def fetch(url: str, ranges: str | None = None, tries: int = 4) -> bytes:
    headers = {"User-Agent": AGENT}
    if ranges:
        headers["Range"] = f"bytes={ranges}"
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code in (403, 404):
                raise
            if attempt == tries - 1:
                raise
        except Exception:
            if attempt == tries - 1:
                raise
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(url)


def exists(url: str) -> bool:
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": AGENT})
        with urllib.request.urlopen(request, timeout=30):
            return True
    except Exception:
        return False


def parallel(function, items, workers=12):
    with cf.ThreadPoolExecutor(workers) as pool:
        return list(pool.map(function, items))


# MARK: - Values to bytes

def encode(layer: str, values: np.ndarray) -> np.ndarray:
    """One byte a point, 255 where there is nothing. The app decodes the same
    way (see `MapCode` in the app)."""
    v = values.astype(np.float32)
    if layer == "rain":          # mm/h, finest near nothing: 0.0016, 0.1 mm/h is 8
        code = np.sqrt(np.clip(v, 0, 100) / 100) * 254
    elif layer == "temperature":  # °C, -50 to 51.6 in steps of 0.4
        code = np.clip((v + 50) / 0.4, 0, 254)
    elif layer == "wind":        # km/h, 0 to 254
        code = np.clip(v, 0, 254)
    else:                        # clouds, %, in steps of 2
        code = np.clip(v, 0, 100) / 2
    out = np.rint(np.nan_to_num(code, nan=0)).astype(np.uint8)
    out[~np.isfinite(v)] = MISSING
    return out


def write_chunks(frames: np.ndarray, folder: str) -> list[str]:
    """`frames` is (time, rows from the south, columns from the west).
    Returns the pieces written, as "row-column"."""
    os.makedirs(folder, exist_ok=True)
    _, rows, cols = frames.shape
    written = []
    for r in range(0, rows, CHUNK):
        for c in range(0, cols, CHUNK):
            block = frames[:, r:r + CHUNK, c:c + CHUNK]
            if (block == MISSING).all():
                continue
            delta = block.copy()
            delta[1:] = block[1:] - block[:-1]        # wraps, as the app's sum does
            packer = zlib.compressobj(9, zlib.DEFLATED, -15)
            data = packer.compress(np.ascontiguousarray(delta).tobytes()) + packer.flush()
            name = f"{r // CHUNK}-{c // CHUNK}"
            with open(os.path.join(folder, name + ".bin"), "wb") as out:
                out.write(data)
            written.append(name)
    return written


# MARK: - GRIB

def grib_grid(data: bytes):
    """A regular latitude-longitude GRIB field as (values from the south and
    the west, first latitude, first longitude, spacing)."""
    handle = eccodes.codes_new_from_message(data)
    try:
        ni, nj = eccodes.codes_get(handle, "Ni"), eccodes.codes_get(handle, "Nj")
        values = eccodes.codes_get_values(handle).astype(np.float32)
        if eccodes.codes_get(handle, "bitmapPresent"):
            values[values == eccodes.codes_get(handle, "missingValue")] = np.nan
        grid = values.reshape(nj, ni)
        lat_first = eccodes.codes_get(handle, "latitudeOfFirstGridPointInDegrees")
        lat_last = eccodes.codes_get(handle, "latitudeOfLastGridPointInDegrees")
        lon_first = eccodes.codes_get(handle, "longitudeOfFirstGridPointInDegrees")
        step = eccodes.codes_get(handle, "iDirectionIncrementInDegrees")
        if not eccodes.codes_get(handle, "jScansPositively"):
            grid = grid[::-1]
        lat0 = min(lat_first, lat_last)
        lon0 = lon_first - 360 if lon_first > 180 else lon_first
        return grid, lat0, lon0, step
    finally:
        eccodes.codes_release(handle)


# MARK: - Forecast models

class Model:
    id = ""
    credit = ""
    spacing = 0.0

    def runs(self, now: dt.datetime) -> list[dt.datetime]:
        raise NotImplementedError

    def ready(self, run: dt.datetime) -> bool:
        raise NotImplementedError

    def build(self, run: dt.datetime) -> tuple[dict, dict[str, np.ndarray]]:
        """The grid and, per layer, (hours, rows, columns) of bytes."""
        raise NotImplementedError


class Icon(Model):
    """DWD's ICON-D2 or ICON-EU, on their regular latitude-longitude grids."""

    def __init__(self, id, credit, spacing, base, every, steps, name):
        self.id, self.credit, self.spacing = id, credit, spacing
        self.base, self.every, self.steps, self.name = base, every, steps, name

    def runs(self, now):
        latest = now.replace(minute=0, second=0, microsecond=0)
        latest -= dt.timedelta(hours=latest.hour % self.every)
        return [latest - dt.timedelta(hours=self.every * k) for k in range(4)]

    def url(self, run, variable, step):
        return f"{self.base}/{run:%H}/{variable}/" + self.name(run, variable, step)

    def ready(self, run):
        return exists(self.url(run, "tot_prec", self.steps))

    def build(self, run):
        steps = list(range(min(self.steps, HOURS) + 1))

        def field(job):
            variable, step = job
            grid, lat0, lon0, spacing = grib_grid(bz2.decompress(fetch(self.url(run, variable, step))))
            return job, grid, (lat0, lon0, spacing)

        jobs = [(v, s) for s in steps for v in ("t_2m", "tot_prec", "u_10m", "v_10m", "clct")]
        fields = {}
        frame = None
        for job, grid, where in parallel(field, jobs, workers=16):
            fields[job] = grid
            frame = where
        lat0, lon0, spacing = frame
        rows, cols = fields[("t_2m", 0)].shape

        # Hour h is what falls in the hour up to it; the first takes the second's.
        out = {layer: np.full((len(steps), rows, cols), MISSING, np.uint8) for layer in LAYERS}
        for s in steps:
            rain = fields[("tot_prec", max(s, 1))] - fields[("tot_prec", max(s, 1) - 1)]
            out["rain"][s] = encode("rain", np.maximum(rain, 0))
            out["temperature"][s] = encode("temperature", fields[("t_2m", s)] - 273.15)
            out["wind"][s] = encode("wind", np.hypot(fields[("u_10m", s)], fields[("v_10m", s)]) * 3.6)
            out["clouds"][s] = encode("clouds", fields[("clct", s)])
        grid = dict(lat0=round(lat0, 4), lon0=round(lon0, 4), step=spacing, rows=rows, cols=cols,
                    start=int(run.timestamp()))
        return grid, out


class Gfs(Model):
    """NOAA's GFS at a quarter degree, each field read on its own from the
    big files by the byte ranges their index gives."""

    id, credit, spacing = "gfs", "NOAA GFS", 0.25
    base = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
    fields = {"TMP": "2 m above ground", "UGRD": "10 m above ground", "VGRD": "10 m above ground",
              "PRATE": "surface", "TCDC": "entire atmosphere"}

    def runs(self, now):
        latest = now.replace(minute=0, second=0, microsecond=0)
        latest -= dt.timedelta(hours=latest.hour % 6)
        return [latest - dt.timedelta(hours=6 * k) for k in range(4)]

    def url(self, run, step):
        return f"{self.base}/gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f{step:03d}"

    def ready(self, run):
        return exists(self.url(run, HOURS + 1) + ".idx")

    def build(self, run):
        steps = list(range(1, HOURS + 2))

        def hour(step):
            lines = fetch(self.url(run, step) + ".idx").decode().splitlines()
            entries = [line.split(":") for line in lines]
            found = {}
            for i, entry in enumerate(entries):
                name, level, kind = entry[3], entry[4], entry[5]
                if self.fields.get(name) == level and "ave" not in kind and "acc" not in kind and name not in found:
                    end = int(entries[i + 1][1]) - 1 if i + 1 < len(entries) else ""
                    found[name] = f"{entry[1]}-{end}"
            grids = {name: grib_grid(fetch(self.url(run, step), ranges))[0] for name, ranges in found.items()}
            return step, grids

        rows, cols = 721, 1440
        out = {layer: np.full((len(steps), rows, cols), MISSING, np.uint8) for layer in LAYERS}
        for step, grids in parallel(hour, steps, workers=12):
            i = step - 1
            # From 0..359.75 east to -180..179.75.
            g = {name: np.roll(grid, cols // 2, axis=1) for name, grid in grids.items()}
            out["temperature"][i] = encode("temperature", g["TMP"] - 273.15)
            out["wind"][i] = encode("wind", np.hypot(g["UGRD"], g["VGRD"]) * 3.6)
            out["rain"][i] = encode("rain", g["PRATE"] * 3600)
            out["clouds"][i] = encode("clouds", g["TCDC"])
        grid = dict(lat0=-90.0, lon0=-180.0, step=0.25, rows=rows, cols=cols,
                    start=int((run + dt.timedelta(hours=1)).timestamp()))
        return grid, out


def overview(grid: dict, layers: dict[str, np.ndarray], factor: int) -> tuple[dict, dict]:
    """The same forecast `factor` times coarser, for the world seen whole.
    Averaged in values, not in bytes, except rain, whose bytes are a square
    root and averaged close enough."""
    out = {}
    rows = (grid["rows"] // factor) * factor
    cols = (grid["cols"] // factor) * factor
    for layer, frames in layers.items():
        block = frames[:, :rows, :cols].reshape(frames.shape[0], rows // factor, factor, cols // factor, factor)
        values = block.astype(np.float32)
        values[block == MISSING] = np.nan
        with np.errstate(all="ignore"):
            mean = np.nanmean(values, axis=(2, 4))
        coded = np.rint(np.nan_to_num(mean, nan=MISSING)).astype(np.uint8)
        out[layer] = coded
    half = grid["step"] * (factor - 1) / 2
    coarse = dict(grid, lat0=grid["lat0"] + half, lon0=grid["lon0"] + half, step=grid["step"] * factor,
                  rows=rows // factor, cols=cols // factor)
    return coarse, out


MODELS = [
    Icon("icon-d2", "DWD ICON-D2", 0.02, "https://opendata.dwd.de/weather/nwp/icon-d2/grib", 3, 48,
         lambda run, v, s: f"icon-d2_germany_regular-lat-lon_single-level_{run:%Y%m%d%H}_{s:03d}_2d_{v}.grib2.bz2"),
    Icon("icon-eu", "DWD ICON-EU", 0.0625, "https://opendata.dwd.de/weather/nwp/icon-eu/grib", 6, HOURS,
         lambda run, v, s: f"icon-eu_europe_regular-lat-lon_single-level_{run:%Y%m%d%H}_{s:03d}_{v.upper()}.grib2.bz2"),
    Gfs(),
]


def update_forecast(state: str, now: dt.datetime, limit: int) -> None:
    """Builds the newest run of each model that has a newer one out, at most
    `limit` of them, so one run of this script stays short."""
    folder = os.path.join(state, "forecast")
    os.makedirs(folder, exist_ok=True)
    built = 0
    for model in MODELS:
        meta_path = os.path.join(folder, model.id + ".json")
        current = json.load(open(meta_path)) if os.path.exists(meta_path) else None
        for run in model.runs(now):
            if current and current["run"] >= int(run.timestamp()):
                break
            if not model.ready(run):
                continue
            if built >= limit:
                log(model.id, "has", f"{run:%d %H}Z", "waiting for the next run of this script")
                break
            log(model.id, "building", f"{run:%Y-%m-%d %H}Z")
            started = time.time()
            grid, layers = model.build(run)
            write_domain(folder, model.id, model.credit, run, grid, layers, current)
            if isinstance(model, Gfs):
                coarse, coarse_layers = overview(grid, layers, 4)
                old = os.path.join(folder, "gfs-world.json")
                write_domain(folder, "gfs-world", model.credit, run, coarse, coarse_layers,
                             json.load(open(old)) if os.path.exists(old) else None)
            log(model.id, "done in", int(time.time() - started), "s")
            built += 1
            break


def write_domain(folder, id, credit, run, grid, layers, previous):
    """Writes a run's pieces. The run before stays published until the next
    one, for an app still holding a manifest that names it."""
    version = f"{run:%Y%m%d%H}"
    chunks = {}
    for layer, frames in layers.items():
        chunks[layer] = write_chunks(frames, os.path.join(folder, id, version, layer))
    meta = dict(id=id, credit=credit, run=int(run.timestamp()), path=f"{id}/{version}",
                hours=int(next(iter(layers.values())).shape[0]), chunks=chunks, **grid)
    if previous and previous["path"] != meta["path"]:
        meta["previous"] = previous["path"]
        if previous.get("previous"):
            shutil.rmtree(os.path.join(folder, previous["previous"]), ignore_errors=True)
    with open(os.path.join(folder, id + ".json"), "w") as out:
        json.dump(meta, out)


# MARK: - Radar

class Grid:
    """A regular latitude-longitude grid a radar is put onto."""

    def __init__(self, lat0, lon0, step, rows, cols):
        self.lat0, self.lon0, self.step, self.rows, self.cols = lat0, lon0, step, rows, cols

    def points(self):
        lats = self.lat0 + np.arange(self.rows) * self.step
        lons = self.lon0 + np.arange(self.cols) * self.step
        return np.meshgrid(lons, lats)

    def meta(self):
        return dict(lat0=self.lat0, lon0=self.lon0, step=self.step, rows=self.rows, cols=self.cols)


class Odim:
    """A composite in ODIM HDF5, read onto `grid` by the nearest pixel."""

    def __init__(self, grid: Grid):
        self.grid = grid
        self.lookup = None

    def place(self, file: h5py.File, values: np.ndarray) -> np.ndarray:
        if self.lookup is None:
            where = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in file["where"].attrs.items()}
            projection = pyproj.Proj(where["projdef"])
            x_ll, y_ll = projection(where["LL_lon"], where["LL_lat"])
            x_ur, y_ur = projection(where["UR_lon"], where["UR_lat"])
            lons, lats = self.grid.points()
            x, y = projection(lons, lats)
            col = np.floor((x - x_ll) / where["xscale"]).astype(np.int64)
            row = np.floor((y_ur - y) / where["yscale"]).astype(np.int64)
            inside = (col >= 0) & (col < where["xsize"]) & (row >= 0) & (row < where["ysize"])
            self.lookup = (np.where(inside, row, 0), np.where(inside, col, 0), inside)
        row, col, inside = self.lookup
        out = values[row, col]
        out[~inside] = np.nan
        return out


class MeteoSwiss(Odim):
    id, credit, coarser = "radar-ch", "MeteoSwiss", 8
    stac = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-radar-precip/items"

    def __init__(self):
        super().__init__(Grid(43.6, 2.7, 0.01, 580, 980))
        self.assets = {}

    def prepare(self, now):
        for day in (now, now - dt.timedelta(hours=3)):
            try:
                item = json.loads(fetch(f"{self.stac}/{day:%Y%m%d}-ch"))
            except Exception:
                continue
            for name, asset in item["assets"].items():
                match = re.match(r"rzc(\d\d)(\d{3})(\d{4})", name)
                if match:
                    year, doy, hhmm = match.groups()
                    when = dt.datetime.strptime(f"20{year}{doy}{hhmm}", "%Y%j%H%M").replace(tzinfo=dt.timezone.utc)
                    self.assets[int(when.timestamp())] = asset["href"]

    def latest(self):
        return max(self.assets) if self.assets else None

    def frame(self, stamp):
        href = self.assets.get(stamp)
        if not href:
            return None
        with h5py.File(io.BytesIO(fetch(href)), "r") as file:
            data = file["dataset1/data1/data"][()].astype(np.float32)
            what = file["dataset1/data1/what"].attrs
            data = data * what["gain"] + what["offset"]
            return self.place(file, data)


class Dwd(Odim):
    id, credit, coarser = "radar-de", "DWD", 8
    base = "https://opendata.dwd.de/weather/radar/composite/rv"

    def __init__(self):
        super().__init__(Grid(45.6, 1.4, 0.015, 694, 1160))
        self.listed = set()

    def prepare(self, now):
        page = fetch(self.base + "/").decode()
        for name in re.findall(r'composite_rv_(\d{8}_\d{4})\.tar', page):
            when = dt.datetime.strptime(name, "%Y%m%d_%H%M").replace(tzinfo=dt.timezone.utc)
            self.listed.add(int(when.timestamp()))

    def latest(self):
        return max(self.listed) if self.listed else None

    def frame(self, stamp):
        if stamp not in self.listed:
            return None
        when = dt.datetime.fromtimestamp(stamp, dt.timezone.utc)
        archive = tarfile.open(fileobj=io.BytesIO(fetch(f"{self.base}/composite_rv_{when:%Y%m%d_%H%M}.tar")))
        member = next(m for m in archive.getmembers() if m.name.endswith("_000-hd5"))
        with h5py.File(io.BytesIO(archive.extractfile(member).read()), "r") as file:
            raw = file["dataset1/data1/data"][()]
            what = file["dataset1/data1/what"].attrs
            data = raw.astype(np.float32) * what["gain"] + what["offset"]
            data[raw == what["nodata"]] = np.nan
            data[raw == what["undetect"]] = 0
            # Five minutes' worth, in millimetres: as a rate, twelve times that.
            return self.place(file, np.maximum(data, 0) * 12)


class Mrms:
    """NOAA's MRMS precipitation rate over the United States, a hundredth of
    a degree, averaged three by three."""

    id, credit, coarser = "radar-us", "NOAA MRMS", 5
    base = "https://noaa-mrms-pds.s3.amazonaws.com"
    grid = Grid(20.015, -129.985, 0.03, 1166, 2333)

    def __init__(self):
        self.listed = set()

    def prepare(self, now):
        for day in {now.date(), (now - dt.timedelta(hours=3)).date()}:
            prefix = f"CONUS/PrecipRate_00.00/{day:%Y%m%d}/MRMS_PrecipRate_00.00_{day:%Y%m%d}-"
            token = None
            while True:
                query = f"?list-type=2&prefix={prefix}" + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
                page = fetch(self.base + "/" + query).decode()
                for stamp in re.findall(r"_(\d{8}-\d{6})\.grib2\.gz", page):
                    when = dt.datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=dt.timezone.utc)
                    if when > now - dt.timedelta(hours=3):
                        self.listed.add(int(when.timestamp()))
                more = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", page)
                if not more:
                    break
                token = more.group(1)

    def latest(self):
        return max(self.listed) if self.listed else None

    def frame(self, stamp):
        if stamp not in self.listed:
            return None
        when = dt.datetime.fromtimestamp(stamp, dt.timezone.utc)
        data = gzip.decompress(fetch(f"{self.base}/CONUS/PrecipRate_00.00/{when:%Y%m%d}/"
                                     f"MRMS_PrecipRate_00.00_{when:%Y%m%d-%H%M%S}.grib2.gz"))
        grid, *_ = grib_grid(data)
        grid[grid < 0] = np.nan                       # -3: no radar sees it
        rows, cols = self.grid.rows * 3, self.grid.cols * 3
        block = grid[:rows, :cols].reshape(self.grid.rows, 3, self.grid.cols, 3)
        with np.errstate(all="ignore"):
            return np.nanmean(block, axis=(1, 3))


RADARS = [MeteoSwiss(), Dwd(), Mrms()]


def update_radar(state: str, now: dt.datetime) -> tuple[int, list]:
    """Reads the frames not yet kept, and settles on the newest ten-minute
    mark every radar that is running has reached, so the last frame is
    never missing a country."""
    latest_marks = []
    for radar in RADARS:
        try:
            radar.prepare(now)
        except Exception as error:
            log(radar.id, "listing failed:", error)
        latest = radar.latest()
        if latest and latest > now.timestamp() - 3600:
            latest_marks.append(latest - latest % RADAR_STEP)
    if not latest_marks:
        return 0, []
    end = min(latest_marks)
    times = [end - RADAR_STEP * k for k in reversed(range(RADAR_FRAMES))]

    for radar in RADARS:
        folder = os.path.join(state, "radar", radar.id)
        os.makedirs(folder, exist_ok=True)
        wanted = [t for t in times if not os.path.exists(os.path.join(folder, f"{t}.npy"))]

        def read(stamp, radar=radar, folder=folder):
            try:
                values = radar.frame(stamp)
            except Exception as error:
                log(radar.id, stamp, "failed:", error)
                return
            if values is not None:
                np.save(os.path.join(folder, f"{stamp}.npy"), encode("rain", values))

        # MeteoSwiss and DWD read their files one at a time, the lookup is built on the first.
        if wanted:
            read(wanted[0])
            parallel(read, wanted[1:], workers=6)
        for name in os.listdir(folder):
            if name.endswith(".npy") and int(name[:-4]) < times[0] - 3600:
                os.remove(os.path.join(folder, name))
    return end, times


def write_radar(state: str, site: str, times: list) -> list:
    """Each radar's frames, and the same `coarser` times coarser for the map
    seen from far out. The frames up to ten minutes before go out as well,
    for an app still holding the manifest before this one."""
    domains = []
    earlier = [t - RADAR_STEP for t in times]
    for radar in RADARS:
        folder = os.path.join(state, "radar", radar.id)
        grid = radar.grid
        for stamps, listed in ((times, True), (earlier, False)):
            frames = np.full((len(stamps), grid.rows, grid.cols), MISSING, np.uint8)
            found = 0
            for i, stamp in enumerate(stamps):
                path = os.path.join(folder, f"{stamp}.npy")
                if os.path.exists(path):
                    frames[i] = np.load(path)
                    found += 1
            if not found:
                continue
            version = str(stamps[-1])
            meta = grid.meta()
            coarse, coarse_frames = overview(meta, {"rain": frames}, radar.coarser)
            for id, g, f in ((radar.id, meta, frames), (radar.id + "-wide", coarse, coarse_frames["rain"])):
                chunks = write_chunks(f, os.path.join(site, id, version, "rain"))
                if listed:
                    domains.append(dict(id=id, credit=radar.credit, path=f"{id}/{version}",
                                        times=stamps, chunks={"rain": chunks}, **g))
    return domains


# MARK: - The site

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="state")
    parser.add_argument("--site", default="site")
    parser.add_argument("--models", type=int, default=1, help="forecast runs built at most per run")
    parser.add_argument("--no-radar", action="store_true")
    options = parser.parse_args()

    now = dt.datetime.now(dt.timezone.utc)
    os.makedirs(options.state, exist_ok=True)
    try:
        update_forecast(options.state, now, options.models)
    except Exception as error:
        # A model that fails now is tried again next run; the last one stays.
        log("forecast failed:", repr(error))

    shutil.rmtree(options.site, ignore_errors=True)
    os.makedirs(options.site)
    forecast = []
    folder = os.path.join(options.state, "forecast")
    for model_id in ("icon-d2", "icon-eu", "gfs", "gfs-world"):
        meta_path = os.path.join(folder, model_id + ".json")
        if not os.path.exists(meta_path):
            continue
        meta = json.load(open(meta_path))
        source = os.path.join(folder, meta["path"])
        if os.path.isdir(source):
            for path in (meta["path"], meta.pop("previous", None)):
                if path and os.path.isdir(os.path.join(folder, path)):
                    shutil.copytree(os.path.join(folder, path), os.path.join(options.site, path))
            forecast.append(meta)

    radar = []
    if not options.no_radar:
        try:
            _, times = update_radar(options.state, now)
            if times:
                radar = write_radar(options.state, options.site, times)
        except Exception as error:
            log("radar failed:", repr(error))

    manifest = dict(version=1, updated=int(now.timestamp()), chunk=CHUNK, forecast=forecast, radar=radar)
    with open(os.path.join(options.site, "manifest.json"), "w") as out:
        json.dump(manifest, out, separators=(",", ":"))
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"), options.site)
    open(os.path.join(options.site, ".nojekyll"), "w").close()
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(options.site) for f in fs)
    log("site", f"{size / 1e6:.1f} MB,", len(forecast), "forecast and", len(radar), "radar domains")


if __name__ == "__main__":
    sys.exit(main())
