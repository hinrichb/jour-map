# jour-map

The data behind the weather map in [Jour](https://github.com/hinrichb), rebuilt every ten minutes by GitHub Actions and published with GitHub Pages.

- Forecast, every hour for the next two and a half days: DWD ICON-D2 (2 km, Germany and the Alps), DWD ICON-EU (7 km, Europe), NOAA GFS (25 km, worldwide).
- Radar, the last two hours every ten minutes: MeteoSwiss, DWD, NOAA MRMS.

Layers: rain (mm/h), temperature (°C), wind (km/h), cloud cover (%).

## Format

`manifest.json` lists every domain: its grid (`lat0`, `lon0` of the south-west point, `step` in degrees, `rows`, `cols`), its hours (`start`, `hours`) or radar `times`, and which 128 × 128 pieces exist per layer. A piece is `<path>/<layer>/<row>-<col>.bin`: raw DEFLATE of `frames × rows × cols` bytes, rows from the south, each frame stored as its difference from the one before (mod 256). 255 means no data.

| Layer | Byte |
| --- | --- |
| rain | `sqrt(mm/h / 100) × 254` |
| temperature | `(°C + 50) / 0.4` |
| wind | km/h |
| clouds | `% / 2` |

## Sources

Deutscher Wetterdienst (CC BY 4.0), MeteoSwiss ("Source: MeteoSwiss"), NOAA (public domain).

## Run locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python build.py --state state --site site --models 3
```
