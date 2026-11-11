# External input media

Input media are not redistributed. Download annotations first, then run:

```bash
python vpbench.py fetch-media --data data --tasks B250_135
python vpbench.py fetch-media --data data --check
```

`data/external.json` describes 215 input files and their sources, trim/frame extraction, fingerprints and hashes.
`data/NOTICE.md` lists task-level attributions and terms. Obtain any restricted source dataset under its own terms,
then provide its local root with `--datasets-root /path/to/licensed/datasets`.
Source-specific details are in [source reconstruction](source-reconstruction.md).

`fetch_report.json` distinguishes identical files, accepted reconstructions and unavailable sources. The command returns a nonzero exit status if any source is unavailable, mismatched or low resolution. Inspect the report
and use `--check` before generation.
Online video hosts may restrict access by region or ask servers on datacenter IP addresses to confirm they are not
a robot; run the tool from a regular network if that happens. The tool never logs in, never reads cookies or browser
data, and refuses cookie and credential options. Other yt-dlp options, such as a proxy, can be passed with
`--ytdlp-arg=OPTION`. Proxy addresses and credentials are never embedded in this package.
