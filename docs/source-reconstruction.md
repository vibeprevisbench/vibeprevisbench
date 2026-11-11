# External inputs

**215 files of 148 tasks are not in this repository.** They are the photos, first frames and reference videos of the photo, continuation and video tasks; they belong to third parties, so the dataset gives their sources instead. Put them in place before you run the benchmark. `python vpbench.py fetch-media --data data` attempts reconstruction from the listed sources and records the outcome of every file. `python vpbench.py fetch-media --data data --check` lists missing files. Source availability has not been verified for the full release on an ordinary network. The 94 script tasks need no media.

`external.json` holds the same list in machine-readable form. Each entry below gives the path the file goes to, what it is, where the original comes from, how it was cut, the sha256 of the copy the benchmark was run with and how a download is checked.

## Run

Requirements: Python 3.8+, `ffmpeg` and `ffprobe` 5.1 or newer on `PATH`, a recent `yt-dlp` for YouTube and
Bilibili, and `pip install numpy opencv-python-headless` for the photos: the script replays the photo steps with
OpenCV as they were run, which gives byte-identical photos. Without OpenCV, photos are rebuilt with ffmpeg and match
in content only, and the ScanNet++ photos that need a fisheye undistortion cannot be rebuilt.

```bash
python vpbench.py fetch-media --data data --jobs 4                          # everything
python vpbench.py fetch-media --data data --tasks B250_190,M50_038          # some tasks only
python vpbench.py fetch-media --data data --kinds image                     # photo tasks only
python vpbench.py fetch-media --data data --check                           # report what is missing
```

The script uses the network settings of your environment: downloads honour `HTTP_PROXY` / `HTTPS_PROXY`, and
`--ytdlp-arg=--proxy=URL` passes a proxy to yt-dlp. It never logs in, never reads browser data and refuses cookie or
credential options for yt-dlp. YouTube may ask servers on datacenter IP addresses to confirm they are not a robot;
run the script from a regular network if that happens. Results are written to `fetch_report.json`.

## What "the same file" means

| Source | Files | Check |
|---|---|---|
| Dataset photos (ScanNet++, ARKitScenes, CA-1M, PandaSet, ETH3D) | 101 | The downloaded source file must have the recorded sha256 (ScanNet++, PandaSet, ETH3D); with OpenCV the rebuilt photo is byte-identical to ours (sha256), and every photo is also checked with the fingerprints |
| Fixed files (Internet Archive, NASA, Wikimedia Commons, BMW Group PressClub, Eyepetizer, Sina) | 16 | Same time window, crop, size and frame rate; sha256 of our file, then fingerprints |
| YouTube and Bilibili | 98 | Same time window, crop, size and frame rate; fingerprints of the frame shown every 0.5 s; the source must be served at least at the resolution we used |

Every file is reported as:

- `identical`: byte-identical to the file used in the benchmark;
- `ok`: same content (fingerprints match at the recorded time and size); the bytes differ because the platform or
  the encoder version differs;
- `aligned`: same content after a small time offset or rescale (re-uploaded copy, different intro), found within
  ±3 s (`--search-range`) and applied;
- `lowres`: same content, but the platform served a lower resolution than the one we used; the file is scaled to the
  recorded size and is less sharp;
- `mismatch`: produced, but the content or the source file differs from ours;
- `unavailable`: the original could not be downloaded.

## Photo tasks

The downloader implements automatic retrieval for the listed photo sources. This is not a guarantee of current source availability or access authorization. Some sources may require an authorized local dataset copy supplied with `--datasets-root`.

| Dataset | Items | Source used by the script |
|---|---|---|
| ScanNet++ v2 (DSLR) | 22 items, 52 photos | Members of Hugging Face re-uploads of ScanNet++, pinned to a commit and checked by sha256. If you have the official data (apply at https://scannetpp.mlsg.cit.tum.de/scannetpp/), pass `--datasets-root` with `scannetpp/data/<scene>/dslr/...` under it and it is used first. Use is subject to the ScanNet++ Terms of Use |
| ARKitScenes (vga_wide) | M50_025–029, 25 photos | Range reads of `raw/Validation/<id>/vga_wide.zip` on Apple's public CDN |
| CA-1M (ARKitScenes raw video frames) | 5 items, 11 photos | The public ARKitScenes raw `.mov` on Apple's CDN (0.3–1.5 GB each) |
| PandaSet | 4 items, 9 photos | Members of `pandaset.zip` in the Hugging Face dataset georghess/pandaset (pinned commit) |
| ETH3D | 2 items, 4 photos | Files of the Hugging Face dataset PeterDAI/eth3d (pinned commit) |

Each photo's processing (fisheye undistortion for part of ScanNet++, downscaling for ETH3D, rotation for CA-1M and
ARKitScenes, JPEG settings) is recorded in its `recipe.ops`.

## Known limits

- Uploads can disappear or become private; they are then reported as `unavailable`.
- Without login, Bilibili may serve a lower resolution than the one we used (`lowres`). Bilibili also answers
  HTTP 412 to fast series of requests; the script waits and retries, and a smaller `--jobs` helps.
- The five Eyepetizer videos depend on its public play-URL API.
- `M50_037` was taken from a Sina video URL that has expired; the manifest lists the same film's YouTube upload as an
  alternative, searched over its whole length.
- `B250_170` and `B250_172` take their first frame from public-domain films (Nothing Sacred, 1937; Royal Wedding,
  1951). The script uses public-domain releases on the Internet Archive and searches them within ±600 s of the
  recorded time. Another release has its own transfer and framing, so these two frames match in content, not in bytes.

## When some media cannot be rebuilt

A complete run requires all input files. The downloader accepts `identical`, `ok` and `aligned`; `lowres` must be inspected and reported as degraded input, and is not established as equivalent to the paper inputs.
Otherwise, as for other benchmarks built on web videos, score the tasks whose media were rebuilt, report how many tasks
were scored, and list the missing task ids from `fetch_report.json`; the scorer restricts a run to a list of tasks.
Text tasks (94) need no media and are always complete.

## Removal requests

If you own content referenced here and want it removed, open a discussion in the Community tab of this repository or
contact us through https://vibeprevisbench.github.io. We then remove the reference from the manifest.

## Licence

The manifest and the script record only where each file comes from and how it was cut; they contain no media content.
Every record's `licence` field gives the terms of its source. The media remain the property of their owners: use the
rebuilt files only as those terms allow (typically non-commercial research) and do not redistribute them.

## Files by task

### B250_135

- `tasks/B250_135/media/photo1.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/25f3b7a318/dslr/resized_images/DSC07216.JPG`, source sha256 `e4f2b392af13423bf5f74ee696cc0f978150bb2a4e7d31dc07cc9d18a9173d8e`  
  Source: member `scannetpp_eval/data/25f3b7a318/dslr/resized_images/DSC07216.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `e4f2b392af13423bf5f74ee696cc0f978150bb2a4e7d31dc07cc9d18a9173d8e`  
  Cut: ScanNet++ scene 25f3b7a318, frame DSC07216; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `66a48137c9b4f2f4627a9530b23689021c5d7e603091aa3976fe335e1b7cf91a`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_135/media/photo2.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/25f3b7a318/dslr/resized_images/DSC06859.JPG`, source sha256 `58cf59f1bd5398599de08bf8ca003dd1a1214d35a139ac74a332dcbac231f6f3`  
  Source: member `scannetpp_eval/data/25f3b7a318/dslr/resized_images/DSC06859.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `58cf59f1bd5398599de08bf8ca003dd1a1214d35a139ac74a332dcbac231f6f3`  
  Cut: ScanNet++ scene 25f3b7a318, frame DSC06859; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `bfa1562621b8deaf46159338343671ec7d8887e8cf3dde792ef2fe8a26f5f276`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_135/media/photo3.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/25f3b7a318/dslr/resized_images/DSC07223.JPG`, source sha256 `bb3ba86514fd086de787f368f171909cb453fe68716281cbe86fa49ead8af224`  
  Source: member `scannetpp_eval/data/25f3b7a318/dslr/resized_images/DSC07223.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `bb3ba86514fd086de787f368f171909cb453fe68716281cbe86fa49ead8af224`  
  Cut: ScanNet++ scene 25f3b7a318, frame DSC07223; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `039b474678799ac429b0ef357761c9ebdbc1f5b5a193d7a5ad70e83c93e7671d`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_136

- `tasks/B250_136/media/photo1.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/c4c04e6d6c/dslr/resized_images/DSC03389.JPG`, source sha256 `a242d659077868ad8f48b31fec764e5635e28c6ddb1d6996572038474b1b1508`  
  Source: member `scannetpp_eval/data/c4c04e6d6c/dslr/resized_images/DSC03389.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `a242d659077868ad8f48b31fec764e5635e28c6ddb1d6996572038474b1b1508`  
  Cut: ScanNet++ scene c4c04e6d6c, frame DSC03389; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `cee9b7d2ac8aa7958c658150366f371526b83fe8920b2b9591dbfc3abca8d897`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_136/media/photo2.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/c4c04e6d6c/dslr/resized_images/DSC03021.JPG`, source sha256 `095a064ea4f85e7dda347a28999f38d54ef4a923ce1b9db4559693dd68cb5977`  
  Source: member `scannetpp_eval/data/c4c04e6d6c/dslr/resized_images/DSC03021.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `095a064ea4f85e7dda347a28999f38d54ef4a923ce1b9db4559693dd68cb5977`  
  Cut: ScanNet++ scene c4c04e6d6c, frame DSC03021; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `eb989bc38c8d03898e518652b119f1e74ccfd11acbdc92f0d642b97ca6c135a7`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_136/media/photo3.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/c4c04e6d6c/dslr/resized_images/DSC03386.JPG`, source sha256 `27db86a3152fb688373ad595f189d13e5c39a721da1f8a2e9958ffec165b914f`  
  Source: member `scannetpp_eval/data/c4c04e6d6c/dslr/resized_images/DSC03386.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `27db86a3152fb688373ad595f189d13e5c39a721da1f8a2e9958ffec165b914f`  
  Cut: ScanNet++ scene c4c04e6d6c, frame DSC03386; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `00d37d1c358fad92bcfab55df3671fa75a36fb25bca673ffb2a2a4dfa91f7228`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_137

- `tasks/B250_137/media/photo1.jpg` — dataset photo, 1752×1168, 0.5 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/7831862f02/dslr/resized_images/DSC04252.JPG`, source sha256 `78b8504cbc7d3a320fd7657e1161215f4c3507efb279e1b69dea73baee18fa1b`  
  Source: member `scannetpp_eval/data/7831862f02/dslr/resized_images/DSC04252.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `78b8504cbc7d3a320fd7657e1161215f4c3507efb279e1b69dea73baee18fa1b`  
  Cut: ScanNet++ scene 7831862f02, frame DSC04252; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `a82f8f27539eb5637dbb39d331bbed6d93e205ef3b0cd7964678af1b0eecf07b`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_137/media/photo2.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/7831862f02/dslr/resized_images/DSC04186.JPG`, source sha256 `a4751a19e8657e19c74cab3b8c1c0b3781d6fca910530c1b7c22e24fcbc17f09`  
  Source: member `scannetpp_eval/data/7831862f02/dslr/resized_images/DSC04186.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `a4751a19e8657e19c74cab3b8c1c0b3781d6fca910530c1b7c22e24fcbc17f09`  
  Cut: ScanNet++ scene 7831862f02, frame DSC04186; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `105cb330afa26cd0cb405a31b8308cc5b2f2571bd9932b9e0775ed45dedcca0a`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_137/media/photo3.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/7831862f02/dslr/resized_images/DSC03890.JPG`, source sha256 `610f067acb3502fc37d471a0abd411932ebf0a103d06adaedb5cc96f14cabecd`  
  Source: member `scannetpp_eval/data/7831862f02/dslr/resized_images/DSC03890.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `610f067acb3502fc37d471a0abd411932ebf0a103d06adaedb5cc96f14cabecd`  
  Cut: ScanNet++ scene 7831862f02, frame DSC03890; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `57f68512c7fb99f4588cb83c241ed12a416325ebd1434b409f002a0512756cd7`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_139

- `tasks/B250_139/media/photo1.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/09d6e808b4/dslr/resized_undistorted_images/DSC06092.JPG`, source sha256 `9fcfa007e57adcb21d696476dc865e4a137a72b460624e0cd67bdc72515d8ccc`  
  Source: member `09d6e808b4/dslr/resized_undistorted_images/DSC06092.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0003.zip>, source sha256 `9fcfa007e57adcb21d696476dc865e4a137a72b460624e0cd67bdc72515d8ccc`  
  Cut: ScanNet++ scene 09d6e808b4, frame DSC06092; processing: jpeg  
  sha256 `07b3b48ecb12fb8203acbf70f28b41485fdec7b2905462f48840860581692ce7`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_139/media/photo2.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/09d6e808b4/dslr/resized_undistorted_images/DSC05451.JPG`, source sha256 `04da95ced0093c9038746e8ccb8fd4bdaf8cd91015ba9ec190aca379c774d4e2`  
  Source: member `09d6e808b4/dslr/resized_undistorted_images/DSC05451.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0003.zip>, source sha256 `04da95ced0093c9038746e8ccb8fd4bdaf8cd91015ba9ec190aca379c774d4e2`  
  Cut: ScanNet++ scene 09d6e808b4, frame DSC05451; processing: jpeg  
  sha256 `d8f3b51c666f414adc4496fc60bd563baa1026b2548143fd80abf9883486fe92`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_139/media/photo3.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/09d6e808b4/dslr/resized_undistorted_images/DSC06320.JPG`, source sha256 `7d730bf04582d4ea3e7041ae1af4c21f086be9ea8cfce5424e63d6032897b59b`  
  Source: member `09d6e808b4/dslr/resized_undistorted_images/DSC06320.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0003.zip>, source sha256 `7d730bf04582d4ea3e7041ae1af4c21f086be9ea8cfce5424e63d6032897b59b`  
  Cut: ScanNet++ scene 09d6e808b4, frame DSC06320; processing: jpeg  
  sha256 `b7725a1758ac17ad69a3266881a4e5a03d44ff74db4e58683422e63d9a54a2f4`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_139/media/photo4.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/09d6e808b4/dslr/resized_undistorted_images/DSC05648.JPG`, source sha256 `87b59c014501b92cccfcf1063d2b639ba31ff85610a87eac490fa3ea1d9f1650`  
  Source: member `09d6e808b4/dslr/resized_undistorted_images/DSC05648.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0003.zip>, source sha256 `87b59c014501b92cccfcf1063d2b639ba31ff85610a87eac490fa3ea1d9f1650`  
  Cut: ScanNet++ scene 09d6e808b4, frame DSC05648; processing: jpeg  
  sha256 `b9965d5de0ea09c96659e26287ec994c08638ca15b4d4dafd8ca9400a86b1ef7`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_140

- `tasks/B250_140/media/photo1.jpg` — dataset photo, 2070×1379, 1.0 MB  
  Official data (optional, used first when you pass `--datasets-root`): `eth3d/playground/images/dslr_images_undistorted/DSC_0575.JPG`, source sha256 `b172573898f7abf831b612293fa50a6bd5c908296a7e088e79152fe00a60eb87`  
  Source: <https://huggingface.co/datasets/PeterDAI/eth3d/resolve/fb2ddc77390926c335c48dace97533f519fdd571/playground/images/dslr_images_undistorted/DSC_0575.JPG>, source sha256 `b172573898f7abf831b612293fa50a6bd5c908296a7e088e79152fe00a60eb87`  
  Cut: ETH3D scene playground, frame DSC_0575; processing: resize, jpeg, jpeg  
  sha256 `60f368d1571212bea7d6bdfc11d41c598446b6af56b442e7648e5c87bd8e3488`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ETH3D: CC BY-NC-SA 4.0 (Schops et al., CVPR 2017).

- `tasks/B250_140/media/photo2.jpg` — dataset photo, 2071×1380, 0.9 MB  
  Official data (optional, used first when you pass `--datasets-root`): `eth3d/playground/images/dslr_images_undistorted/DSC_0577.JPG`, source sha256 `41a20df5aa4ff6ceca0c53b9df9b42fa6f5e21618d11b61c0c400d6bb5fccb3f`  
  Source: <https://huggingface.co/datasets/PeterDAI/eth3d/resolve/fb2ddc77390926c335c48dace97533f519fdd571/playground/images/dslr_images_undistorted/DSC_0577.JPG>, source sha256 `41a20df5aa4ff6ceca0c53b9df9b42fa6f5e21618d11b61c0c400d6bb5fccb3f`  
  Cut: ETH3D scene playground, frame DSC_0577; processing: resize, jpeg, jpeg  
  sha256 `5635f29593e1c5d2532d11c9f84b45bedf812c33fa894611a02f91d170462fe7`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ETH3D: CC BY-NC-SA 4.0 (Schops et al., CVPR 2017).

- `tasks/B250_140/media/photo3.jpg` — dataset photo, 2074×1382, 0.8 MB  
  Official data (optional, used first when you pass `--datasets-root`): `eth3d/playground/images/dslr_images_undistorted/DSC_0598.JPG`, source sha256 `d73b50480d49ed8e390602c25ab0c0c981dccbac9b8c14fafdfaef155a4d8e59`  
  Source: <https://huggingface.co/datasets/PeterDAI/eth3d/resolve/fb2ddc77390926c335c48dace97533f519fdd571/playground/images/dslr_images_undistorted/DSC_0598.JPG>, source sha256 `d73b50480d49ed8e390602c25ab0c0c981dccbac9b8c14fafdfaef155a4d8e59`  
  Cut: ETH3D scene playground, frame DSC_0598; processing: resize, jpeg, jpeg  
  sha256 `8ef31bc2cd7d2fa3d6859337b4c62ea94a1e9957f1a964be4c9cb24ee37ff900`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ETH3D: CC BY-NC-SA 4.0 (Schops et al., CVPR 2017).

### B250_141

- `tasks/B250_141/media/photo1.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/06b5863f73/dslr/resized_undistorted_images/02dd3b53_DSC03132.JPG`, source sha256 `e52c4ec8f1a9ef9ccfd309a7cb50d38cba164d1d9c181c00f13c248c2ca0cd6a`  
  Source: member `06b5863f73/dslr/resized_undistorted_images/02dd3b53_DSC03132.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0002.zip>, source sha256 `e52c4ec8f1a9ef9ccfd309a7cb50d38cba164d1d9c181c00f13c248c2ca0cd6a`  
  Cut: ScanNet++ scene 06b5863f73, frame 02dd3b53_DSC03132; processing: jpeg  
  sha256 `727d9cd03e3b19a3fc438ce7cbc17beb3dc803d2cfbeedbec2657661705f0569`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_141/media/photo2.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/06b5863f73/dslr/resized_undistorted_images/02dd3b53_DSC03189.JPG`, source sha256 `9578eceaa7336068ea0f414a4129878f03703a8f294e111573f50114b99702cf`  
  Source: member `06b5863f73/dslr/resized_undistorted_images/02dd3b53_DSC03189.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0002.zip>, source sha256 `9578eceaa7336068ea0f414a4129878f03703a8f294e111573f50114b99702cf`  
  Cut: ScanNet++ scene 06b5863f73, frame 02dd3b53_DSC03189; processing: jpeg  
  sha256 `959a42ce2b670147816ea6def14157cf39e7b6d2c1f7082d3c2b862d941763db`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_141/media/photo3.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/06b5863f73/dslr/resized_undistorted_images/02dd3b53_DSC03201.JPG`, source sha256 `7beedc498664ac9cc7851dba9e801560e8c72167da680572958702612eb6064a`  
  Source: member `06b5863f73/dslr/resized_undistorted_images/02dd3b53_DSC03201.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0002.zip>, source sha256 `7beedc498664ac9cc7851dba9e801560e8c72167da680572958702612eb6064a`  
  Cut: ScanNet++ scene 06b5863f73, frame 02dd3b53_DSC03201; processing: jpeg  
  sha256 `ad9f7c47d70f479e661d2a8ad00f41c05c98cb6f343d9892f5315b919732e951`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_142

- `tasks/B250_142/media/photo1.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/99532296c7/dslr/resized_undistorted_images/bb8762d5_DSC02109.JPG`, source sha256 `36bfc8f25caf03c257a545449059967ea263986649ca2201d40ff42551346028`  
  Source: member `99532296c7/dslr/resized_undistorted_images/bb8762d5_DSC02109.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0006.zip>, source sha256 `36bfc8f25caf03c257a545449059967ea263986649ca2201d40ff42551346028`  
  Cut: ScanNet++ scene 99532296c7, frame bb8762d5_DSC02109; processing: jpeg  
  sha256 `c17345a4216cdde0ea7ada7c1518a02feb9cd577bd6746fb18543d7c9dffaf42`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_142/media/photo2.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/99532296c7/dslr/resized_undistorted_images/bb8762d5_DSC03936.JPG`, source sha256 `079f179a2fcbf01b9128f47f594117b5e3396c518ab4021ee285e59b0df7bc7e`  
  Source: member `99532296c7/dslr/resized_undistorted_images/bb8762d5_DSC03936.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0006.zip>, source sha256 `079f179a2fcbf01b9128f47f594117b5e3396c518ab4021ee285e59b0df7bc7e`  
  Cut: ScanNet++ scene 99532296c7, frame bb8762d5_DSC03936; processing: jpeg  
  sha256 `f7616b0c6a23edf4a0c1367e192f1e769bed7343ee7e96088a58332acc7c46b6`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_142/media/photo3.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/99532296c7/dslr/resized_undistorted_images/bb8762d5_DSC04635.JPG`, source sha256 `1e50b316ca8969c1d0d7a411d1a864bad7fbd07fa15cadc607f7aebd8d592502`  
  Source: member `99532296c7/dslr/resized_undistorted_images/bb8762d5_DSC04635.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0006.zip>, source sha256 `1e50b316ca8969c1d0d7a411d1a864bad7fbd07fa15cadc607f7aebd8d592502`  
  Cut: ScanNet++ scene 99532296c7, frame bb8762d5_DSC04635; processing: jpeg  
  sha256 `52935f4a3045ccfff3ab47535d73867176cfa1b8eb446b36dc4c39a6ad634d43`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_143

- `tasks/B250_143/media/photo1.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/a7631adbe7/dslr/resized_undistorted_images/DSC00901.JPG`, source sha256 `867e5eda7206e42d8fdead812854a4155643af942248d16e3345e5170e45d445`  
  Source: member `a7631adbe7/dslr/resized_undistorted_images/DSC00901.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0007.zip>, source sha256 `867e5eda7206e42d8fdead812854a4155643af942248d16e3345e5170e45d445`  
  Cut: ScanNet++ scene a7631adbe7, frame DSC00901; processing: jpeg  
  sha256 `deb65a63a31b5a2a1114ff0519bd7d1f772abdb7bc15cc707008e3511698eef0`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_143/media/photo2.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/a7631adbe7/dslr/resized_undistorted_images/DSC00387.JPG`, source sha256 `8ee3f96a563b0b7d014aac20effccf59dbd9d285f1adc61c08f33635f04d0e15`  
  Source: member `a7631adbe7/dslr/resized_undistorted_images/DSC00387.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0007.zip>, source sha256 `8ee3f96a563b0b7d014aac20effccf59dbd9d285f1adc61c08f33635f04d0e15`  
  Cut: ScanNet++ scene a7631adbe7, frame DSC00387; processing: jpeg  
  sha256 `6a10ed95712cdf35387632b9fc61dd79b3b4abdaaab303eb4fbefc190d43c1bd`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_143/media/photo3.jpg` — dataset photo, 1752×1168, 0.2 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/a7631adbe7/dslr/resized_undistorted_images/DSC01119.JPG`, source sha256 `3d8a4acfb69d4eb982a2c4183d1565a1051033b8fadf19c5cc968fff2a299281`  
  Source: member `a7631adbe7/dslr/resized_undistorted_images/DSC01119.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0007.zip>, source sha256 `3d8a4acfb69d4eb982a2c4183d1565a1051033b8fadf19c5cc968fff2a299281`  
  Cut: ScanNet++ scene a7631adbe7, frame DSC01119; processing: jpeg  
  sha256 `cae3445e3b4ac0261cfd3001b71edc83f25edbf97ee351cf6def4522721b5d66`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_144

- `tasks/B250_144/media/photo1.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/9c7b4394af/dslr/resized_undistorted_images/DSC02413.JPG`, source sha256 `62e97a632185dac6a96f36cc7f682da40d525709d0071f17e8ddce8865bffcd5`  
  Source: member `9c7b4394af/dslr/resized_undistorted_images/DSC02413.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0003.zip>, source sha256 `62e97a632185dac6a96f36cc7f682da40d525709d0071f17e8ddce8865bffcd5`  
  Cut: ScanNet++ scene 9c7b4394af, frame DSC02413; processing: jpeg  
  sha256 `c584d263b37897d1d1f9dea669314ae0b5aa40238d741448b6f6a7165c838779`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_144/media/photo2.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/9c7b4394af/dslr/resized_undistorted_images/DSC01631.JPG`, source sha256 `26712f715a15ff2d195e59656667562549afda0634a4ebb9a264db8858a4cf50`  
  Source: member `9c7b4394af/dslr/resized_undistorted_images/DSC01631.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0003.zip>, source sha256 `26712f715a15ff2d195e59656667562549afda0634a4ebb9a264db8858a4cf50`  
  Cut: ScanNet++ scene 9c7b4394af, frame DSC01631; processing: jpeg  
  sha256 `0102b9a15b9843adddee4f5f6d08f03ce37d0957ecc05e53156a6f52a5775fc1`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_144/media/photo3.jpg` — dataset photo, 1752×1168, 0.5 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/9c7b4394af/dslr/resized_undistorted_images/DSC02291.JPG`, source sha256 `188615a3a3625e2f2eb5282ff6ffdb30b1f1c2798224063a7b591c52aa7c9325`  
  Source: member `9c7b4394af/dslr/resized_undistorted_images/DSC02291.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0003.zip>, source sha256 `188615a3a3625e2f2eb5282ff6ffdb30b1f1c2798224063a7b591c52aa7c9325`  
  Cut: ScanNet++ scene 9c7b4394af, frame DSC02291; processing: jpeg  
  sha256 `b84be679eb622bd0b4b05fa2e61b89f5b0f578e28e92ba5a67effe7b83002089`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_145

- `tasks/B250_145/media/photo1.jpg` — dataset photo, 1920×1440, 0.5 MB  
  Source: decode index 3804 of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/45260903/45260903.mov>  
  Cut: CA-1M scene 45260903, frame ca_1581872198708; processing: rot90, jpeg, jpeg  
  sha256 `3f7805d82a09ab37f6ed3ec1b4fb8ed3c8fb0756916b9946f96c8ca47333eb76`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: CA-1M: CC BY-NC-ND 4.0; RGB frames from ARKitScenes under the ARKitScenes license.

- `tasks/B250_145/media/photo2.jpg` — dataset photo, 1920×1440, 0.6 MB  
  Source: decode index 689 of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/45260903/45260903.mov>  
  Cut: CA-1M scene 45260903, frame ca_1529976055041; processing: rot90, jpeg, jpeg  
  sha256 `63afc9e69d50ef29a40caba07f9822faab72c58f288be227e0aa2b2f5f3ee9eb`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: CA-1M: CC BY-NC-ND 4.0; RGB frames from ARKitScenes under the ARKitScenes license.

- `tasks/B250_145/media/photo3.jpg` — dataset photo, 1920×1440, 0.5 MB  
  Source: decode index 2484 of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/45260903/45260903.mov>  
  Cut: CA-1M scene 45260903, frame ca_1559880896208; processing: rot90, jpeg, jpeg  
  sha256 `cef0a6b61af71e3654600adf4eebcf07e690598279ee9f7fee08a380cf2ffb7d`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: CA-1M: CC BY-NC-ND 4.0; RGB frames from ARKitScenes under the ARKitScenes license.

### B250_146

- `tasks/B250_146/media/photo1.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/5748ce6f01/dslr/resized_images/DSC09633.JPG`, source sha256 `dd493acff6280bdfe8f73f5df39113199ed183232bfde69a9ed9a0f79135f2e1`  
  Source: member `scannetpp_eval/data/5748ce6f01/dslr/resized_images/DSC09633.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `dd493acff6280bdfe8f73f5df39113199ed183232bfde69a9ed9a0f79135f2e1`  
  Cut: ScanNet++ scene 5748ce6f01, frame DSC09633; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `15ebddc023e80c3603fcd6bb2748d9d2ab0c55d708fc7f7fec40f2556ec3e95a`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_146/media/photo2.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/5748ce6f01/dslr/resized_images/DSC00259.JPG`, source sha256 `ed993b6834b036cac3f19dfd41a54743a98ec948fbe46f0303cadf8c8b23144d`  
  Source: member `scannetpp_eval/data/5748ce6f01/dslr/resized_images/DSC00259.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `ed993b6834b036cac3f19dfd41a54743a98ec948fbe46f0303cadf8c8b23144d`  
  Cut: ScanNet++ scene 5748ce6f01, frame DSC00259; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `aa01325e398db3474a45a983cba34b5c4f0d8078234fda25c2da4943aa65cdd6`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_146/media/photo3.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/5748ce6f01/dslr/resized_images/DSC00096.JPG`, source sha256 `e3016fc27375fe18e78b1661d40e43e60c949753b14d4596873e80df182af8cf`  
  Source: member `scannetpp_eval/data/5748ce6f01/dslr/resized_images/DSC00096.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `e3016fc27375fe18e78b1661d40e43e60c949753b14d4596873e80df182af8cf`  
  Cut: ScanNet++ scene 5748ce6f01, frame DSC00096; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `ffd915a249fac84358a4351b1c28698464ef25ee6c8b1ecf4296baf5d7ca9ead`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_146/media/photo4.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/5748ce6f01/dslr/resized_images/DSC09167.JPG`, source sha256 `8409a5d936471e7ea40cebefe566e67369d40a01a70219e7004dd0b0aa364756`  
  Source: member `scannetpp_eval/data/5748ce6f01/dslr/resized_images/DSC09167.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `8409a5d936471e7ea40cebefe566e67369d40a01a70219e7004dd0b0aa364756`  
  Cut: ScanNet++ scene 5748ce6f01, frame DSC09167; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `b70a0ad9aa4c824875045c17b77f46c948d565e784c1fc3d93f54672fff04cf3`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_147

- `tasks/B250_147/media/photo1.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/6115eddb86/dslr/resized_images/DSC09042.JPG`, source sha256 `720ac3421d95cca2d02c24f301b8cd6b0e9a44a6b1c25757d3dfd2d1e0fe7bbc`  
  Source: member `scannetpp_eval/data/6115eddb86/dslr/resized_images/DSC09042.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `720ac3421d95cca2d02c24f301b8cd6b0e9a44a6b1c25757d3dfd2d1e0fe7bbc`  
  Cut: ScanNet++ scene 6115eddb86, frame DSC09042; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `c65674429575f186adc377e5cdabf062f444f40b9d86aa4ac10ca7c18af7b600`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_147/media/photo2.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/6115eddb86/dslr/resized_images/DSC09126.JPG`, source sha256 `ee37adc89a988b6432c6a4ccf19d9e0035301957693bd05d8192730afa74ff4b`  
  Source: member `scannetpp_eval/data/6115eddb86/dslr/resized_images/DSC09126.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `ee37adc89a988b6432c6a4ccf19d9e0035301957693bd05d8192730afa74ff4b`  
  Cut: ScanNet++ scene 6115eddb86, frame DSC09126; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `b63ac35eb9925a902aa618f857ab111262a78b3ac910f506e4629d5f1a91de1d`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_147/media/photo3.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/6115eddb86/dslr/resized_images/DSC09325.JPG`, source sha256 `a2f07418fb6d99935f5d1fb1ab932c2b3fa63a9a92ce53569b12eeb61cf52941`  
  Source: member `scannetpp_eval/data/6115eddb86/dslr/resized_images/DSC09325.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `a2f07418fb6d99935f5d1fb1ab932c2b3fa63a9a92ce53569b12eeb61cf52941`  
  Cut: ScanNet++ scene 6115eddb86, frame DSC09325; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `727f6f619aaed4658ee63b26503e7d7560990f67c8c569a10e14f9dbd5d88518`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_148

- `tasks/B250_148/media/photo1.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/c5439f4607/dslr/resized_images/DSC05816.JPG`, source sha256 `f4e1732ac0b8dbd7b9d0a817a03568d500279e0313e76bfc293b1470ea4fb546`  
  Source: member `scannetpp_eval/data/c5439f4607/dslr/resized_images/DSC05816.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `f4e1732ac0b8dbd7b9d0a817a03568d500279e0313e76bfc293b1470ea4fb546`  
  Cut: ScanNet++ scene c5439f4607, frame DSC05816; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `eb14aac0a6797c002e02c6f1a5fafb4b660915014517dde4784005b1ab876888`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_148/media/photo2.jpg` — dataset photo, 1752×1168, 0.5 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/c5439f4607/dslr/resized_images/DSC06241.JPG`, source sha256 `d5ef2d20ea014e7d4ef94cfd0ea1b850fd0fb48106dc09111d3894ff2d85120b`  
  Source: member `scannetpp_eval/data/c5439f4607/dslr/resized_images/DSC06241.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `d5ef2d20ea014e7d4ef94cfd0ea1b850fd0fb48106dc09111d3894ff2d85120b`  
  Cut: ScanNet++ scene c5439f4607, frame DSC06241; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `eb2cada2d575a36f15b317981f4b7aa21879b66bd481c70ce2403e09b5489fc2`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_148/media/photo3.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/c5439f4607/dslr/resized_images/DSC06245.JPG`, source sha256 `1c4c68bda6a27fc9736f9687eb3ed52c7cc6d688e45fe3dcc2aec40fccd5bba5`  
  Source: member `scannetpp_eval/data/c5439f4607/dslr/resized_images/DSC06245.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `1c4c68bda6a27fc9736f9687eb3ed52c7cc6d688e45fe3dcc2aec40fccd5bba5`  
  Cut: ScanNet++ scene c5439f4607, frame DSC06245; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `167101a731cfe13ae8fca14f6c9e4c4775125791588f5baaf269b20cf15851c5`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_149

- `tasks/B250_149/media/photo1.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/21d970d8de/dslr/resized_images/DSC09753.JPG`, source sha256 `a961b5040c3976c39b0255d0951f3ed35803ca2fb5b8e65e6a29ba0978bb3af8`  
  Source: member `scannetpp_eval/data/21d970d8de/dslr/resized_images/DSC09753.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `a961b5040c3976c39b0255d0951f3ed35803ca2fb5b8e65e6a29ba0978bb3af8`  
  Cut: ScanNet++ scene 21d970d8de, frame DSC09753; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `2851baa7052df6c51272dd3e5083f3c6c2ba8244db6aa2b45fd22944d4c8e201`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_149/media/photo2.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/21d970d8de/dslr/resized_images/DSC09845.JPG`, source sha256 `6514de6edd653af3ca13637662d44c86551f03ec11e06e88f00f29c343e56b5a`  
  Source: member `scannetpp_eval/data/21d970d8de/dslr/resized_images/DSC09845.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `6514de6edd653af3ca13637662d44c86551f03ec11e06e88f00f29c343e56b5a`  
  Cut: ScanNet++ scene 21d970d8de, frame DSC09845; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `3ca2ad64d82ad12e7e2139266ed2a2d6a6908079be56cae02f148e4d4b167ae2`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_149/media/photo3.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/21d970d8de/dslr/resized_images/DSC09946.JPG`, source sha256 `67752db0df59b2d9bc0a67cae66138047ea1cce2452293d88be6727ca0e46baf`  
  Source: member `scannetpp_eval/data/21d970d8de/dslr/resized_images/DSC09946.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `67752db0df59b2d9bc0a67cae66138047ea1cce2452293d88be6727ca0e46baf`  
  Cut: ScanNet++ scene 21d970d8de, frame DSC09946; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `5506bec2b346859f2ada41886bce1417687b5612976c92b63d60eb67bb24efd5`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_150

- `tasks/B250_150/media/photo1.jpg` — dataset photo, 1752×1168, 0.5 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/3caf4324fd/dslr/resized_undistorted_images/852b663a_DSC05372.JPG`, source sha256 `448d1ec34ff36971264e0286e73315379f840f9c79b539f734ffd75554ede671`  
  Source: member `3caf4324fd/dslr/resized_undistorted_images/852b663a_DSC05372.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0001.zip>, source sha256 `448d1ec34ff36971264e0286e73315379f840f9c79b539f734ffd75554ede671`  
  Cut: ScanNet++ scene 3caf4324fd, frame 852b663a_DSC05372; processing: jpeg  
  sha256 `63e7b44fca948688a0f1037ea96e11f9399e9230e540a8fc08329eea7ad678c9`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_150/media/photo2.jpg` — dataset photo, 1752×1168, 0.5 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/3caf4324fd/dslr/resized_undistorted_images/852b663a_DSC05117.JPG`, source sha256 `7feb04b03444aaa204f44bbef47e6d32b2c062f982e1cfc7e3d43738b7dbf2c4`  
  Source: member `3caf4324fd/dslr/resized_undistorted_images/852b663a_DSC05117.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0001.zip>, source sha256 `7feb04b03444aaa204f44bbef47e6d32b2c062f982e1cfc7e3d43738b7dbf2c4`  
  Cut: ScanNet++ scene 3caf4324fd, frame 852b663a_DSC05117; processing: jpeg  
  sha256 `e532cd27a1550cf4f968be3f2324588e77e41418953e15a97dae3957aeca9f90`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_150/media/photo3.jpg` — dataset photo, 1752×1168, 0.5 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/3caf4324fd/dslr/resized_undistorted_images/dc974236_DSC03995.JPG`, source sha256 `f1bbfff286a262fbd4373c1654b8f990a2dbc1089749f434fb98a1032cf5d13e`  
  Source: member `3caf4324fd/dslr/resized_undistorted_images/dc974236_DSC03995.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0001.zip>, source sha256 `f1bbfff286a262fbd4373c1654b8f990a2dbc1089749f434fb98a1032cf5d13e`  
  Cut: ScanNet++ scene 3caf4324fd, frame dc974236_DSC03995; processing: jpeg  
  sha256 `1fee6b6f5879f269f9e72a30683d56c04665f3ac2c3c3bd452ad687b1e2d5c2e`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_151

- `tasks/B250_151/media/photo1.jpg` — dataset photo, 1752×1168, 0.6 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/5f99900f09/dslr/resized_images/DSC07253.JPG`, source sha256 `180b241c1b8c28705a456ba535103df3231c83bc3c444f2ccc961514197eff4f`  
  Source: member `scannetpp_eval/data/5f99900f09/dslr/resized_images/DSC07253.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `180b241c1b8c28705a456ba535103df3231c83bc3c444f2ccc961514197eff4f`  
  Cut: ScanNet++ scene 5f99900f09, frame DSC07253; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `73eb17183b0c9822607deeea360e1988c01e4f5b43f9ee8a070ae9fb33d4b613`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_151/media/photo2.jpg` — dataset photo, 1752×1168, 0.6 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/5f99900f09/dslr/resized_images/DSC06169.JPG`, source sha256 `52a94aa645c019a13ee0a9a09eaa23dd0cad4375d84f55784bc650d0e7e5b9ce`  
  Source: member `scannetpp_eval/data/5f99900f09/dslr/resized_images/DSC06169.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `52a94aa645c019a13ee0a9a09eaa23dd0cad4375d84f55784bc650d0e7e5b9ce`  
  Cut: ScanNet++ scene 5f99900f09, frame DSC06169; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `5336c96c501833f6d1d241f2cd3c54d1a2bbb0c0d01740f761c2af8cc05db3e8`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

- `tasks/B250_151/media/photo3.jpg` — dataset photo, 1752×1168, 0.5 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/5f99900f09/dslr/resized_images/DSC05451.JPG`, source sha256 `55e1b6fe951c6408c1daf02cafbb43ded82508eaefa029098e53cc24d063ca1a`  
  Source: member `scannetpp_eval/data/5f99900f09/dslr/resized_images/DSC05451.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `55e1b6fe951c6408c1daf02cafbb43ded82508eaefa029098e53cc24d063ca1a`  
  Cut: ScanNet++ scene 5f99900f09, frame DSC05451; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `dc9b631053f59bcd3b0a99e29661d31131b33e019bece2042b827dbdd97aa1a3`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_152

- `tasks/B250_152/media/photo1.jpg` — dataset photo, 1920×1080, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `pandaset/005/camera/left_camera/60.jpg`, source sha256 `6c3017f6bebff61b5c0e6b9b94e0387844e5d2648f7a31387ccd82c5f38b91da`  
  Source: member `pandaset/005/camera/left_camera/60.jpg` of <https://huggingface.co/datasets/georghess/pandaset/resolve/e2e123aea3b3132c67f4b395ec6120f63e190271/pandaset.zip>, source sha256 `6c3017f6bebff61b5c0e6b9b94e0387844e5d2648f7a31387ccd82c5f38b91da`  
  Cut: PandaSet scene 005, frame left_camera_60; processing: jpeg  
  sha256 `1f273ff27c6b2191aba77d5246c116eb3a414284cbc2385b5e668a1c5f83f9d0`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: PandaSet Terms of Use (CC BY 4.0 with additional terms).

- `tasks/B250_152/media/photo2.jpg` — dataset photo, 1920×1080, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `pandaset/005/camera/left_camera/70.jpg`, source sha256 `1043d2777b1e5fe2e9f27446369fdd684a396be3534a8e215b50b7c738c0a03f`  
  Source: member `pandaset/005/camera/left_camera/70.jpg` of <https://huggingface.co/datasets/georghess/pandaset/resolve/e2e123aea3b3132c67f4b395ec6120f63e190271/pandaset.zip>, source sha256 `1043d2777b1e5fe2e9f27446369fdd684a396be3534a8e215b50b7c738c0a03f`  
  Cut: PandaSet scene 005, frame left_camera_70; processing: jpeg  
  sha256 `b87221ffe9c95ab1574f3cd62eb0ea24ce2b51d0063a1bd81ca6c4708fd39c82`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: PandaSet Terms of Use (CC BY 4.0 with additional terms).

- `tasks/B250_152/media/photo3.jpg` — dataset photo, 1920×1080, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `pandaset/005/camera/front_left_camera/45.jpg`, source sha256 `f2ddaa37a5022144eb631a215fe9cfb54e0a50abb61356b6416dae47bbe800fd`  
  Source: member `pandaset/005/camera/front_left_camera/45.jpg` of <https://huggingface.co/datasets/georghess/pandaset/resolve/e2e123aea3b3132c67f4b395ec6120f63e190271/pandaset.zip>, source sha256 `f2ddaa37a5022144eb631a215fe9cfb54e0a50abb61356b6416dae47bbe800fd`  
  Cut: PandaSet scene 005, frame front_left_camera_45; processing: jpeg  
  sha256 `83703cec326207791777df68f1ec4c00c5cbdf56cd06cfb50d1213b32fddeeff`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: PandaSet Terms of Use (CC BY 4.0 with additional terms).

- `tasks/B250_152/media/photo4.jpg` — dataset photo, 1920×1080, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `pandaset/005/camera/front_left_camera/50.jpg`, source sha256 `b04588a18839b8dfd98aaa48107d0844677a722a5d64ab193143cd6d30549248`  
  Source: member `pandaset/005/camera/front_left_camera/50.jpg` of <https://huggingface.co/datasets/georghess/pandaset/resolve/e2e123aea3b3132c67f4b395ec6120f63e190271/pandaset.zip>, source sha256 `b04588a18839b8dfd98aaa48107d0844677a722a5d64ab193143cd6d30549248`  
  Cut: PandaSet scene 005, frame front_left_camera_50; processing: jpeg  
  sha256 `4a06d454562a59afeb8b7da408d23948546d8f3733870a5d4d82dd80b11dd9a4`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: PandaSet Terms of Use (CC BY 4.0 with additional terms).

### B250_153

- `tasks/B250_153/media/photo1.jpg` — dataset photo, 1920×1080, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `pandaset/050/camera/front_right_camera/15.jpg`, source sha256 `14441866606408b81b3cd49f3b5d43dae0a7d8223a744aad446ebca7fc92af6d`  
  Source: member `pandaset/050/camera/front_right_camera/15.jpg` of <https://huggingface.co/datasets/georghess/pandaset/resolve/e2e123aea3b3132c67f4b395ec6120f63e190271/pandaset.zip>, source sha256 `14441866606408b81b3cd49f3b5d43dae0a7d8223a744aad446ebca7fc92af6d`  
  Cut: PandaSet scene 050, frame front_right_camera_15; processing: jpeg  
  sha256 `c9693bf429529233588bafd34dd85c5642624bf1137a59f9383aad1092f58873`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: PandaSet Terms of Use (CC BY 4.0 with additional terms).

- `tasks/B250_153/media/photo2.jpg` — dataset photo, 1920×1080, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `pandaset/050/camera/front_right_camera/25.jpg`, source sha256 `5ca1f11b3c998e9ed6ac1e2a7a6a7d87b078cabebbc15bdb9d243c83608e16f3`  
  Source: member `pandaset/050/camera/front_right_camera/25.jpg` of <https://huggingface.co/datasets/georghess/pandaset/resolve/e2e123aea3b3132c67f4b395ec6120f63e190271/pandaset.zip>, source sha256 `5ca1f11b3c998e9ed6ac1e2a7a6a7d87b078cabebbc15bdb9d243c83608e16f3`  
  Cut: PandaSet scene 050, frame front_right_camera_25; processing: jpeg  
  sha256 `5ac900948e8039fb3d762cd05f56a48fa29c0ab5994e343c2c30e1a4a6130203`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: PandaSet Terms of Use (CC BY 4.0 with additional terms).

- `tasks/B250_153/media/photo3.jpg` — dataset photo, 1920×1080, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `pandaset/050/camera/front_right_camera/35.jpg`, source sha256 `2ca341ebfc77797eb0ef9dad5b6dce41349dc291dd3180c8d464a5fd71071858`  
  Source: member `pandaset/050/camera/front_right_camera/35.jpg` of <https://huggingface.co/datasets/georghess/pandaset/resolve/e2e123aea3b3132c67f4b395ec6120f63e190271/pandaset.zip>, source sha256 `2ca341ebfc77797eb0ef9dad5b6dce41349dc291dd3180c8d464a5fd71071858`  
  Cut: PandaSet scene 050, frame front_right_camera_35; processing: jpeg  
  sha256 `2d08e9be33e6832df6eaba596904f7d9c54f8e2ca17501dc38668ee136928e3f`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: PandaSet Terms of Use (CC BY 4.0 with additional terms).

### B250_154

- `tasks/B250_154/media/photo1.jpg` — dataset photo, 1920×1440, 0.5 MB  
  Source: decode index 672 of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/43828368/43828368.mov>  
  Cut: CA-1M scene 43828368, frame ca_92119929578083; processing: rot90, jpeg, jpeg  
  sha256 `2cb3572efeee5f13fdc0b3c995de012b8e7a71c9fffbcc4a8019de032e80f7bf`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: CA-1M: CC BY-NC-ND 4.0; RGB frames from ARKitScenes under the ARKitScenes license.

- `tasks/B250_154/media/photo2.jpg` — dataset photo, 1920×1440, 0.6 MB  
  Source: decode index 6398 of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/43828368/43828368.mov>  
  Cut: CA-1M scene 43828368, frame ca_92215323411083; processing: rot90, jpeg, jpeg  
  sha256 `1a7ef908634c297f50181caaf15b6548e703638cb45ca720b09424b62e7a0d93`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: CA-1M: CC BY-NC-ND 4.0; RGB frames from ARKitScenes under the ARKitScenes license.

- `tasks/B250_154/media/photo3.jpg` — dataset photo, 1920×1440, 0.6 MB  
  Source: decode index 6764 of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/43828368/43828368.mov>  
  Cut: CA-1M scene 43828368, frame ca_92221420888333; processing: rot90, jpeg, jpeg  
  sha256 `e335ba7ba6da7504f6dbf84903869e026f11489e20f8d8a45bf3efa72ea0732c`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: CA-1M: CC BY-NC-ND 4.0; RGB frames from ARKitScenes under the ARKitScenes license.

- `tasks/B250_154/media/photo4.jpg` — dataset photo, 1920×1440, 0.5 MB  
  Source: decode index 275 of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/43828368/43828368.mov>  
  Cut: CA-1M scene 43828368, frame ca_92113315653000; processing: rot90, jpeg, jpeg  
  sha256 `2b57c67995bc3465f7595b577cad8dc5eb4ac898fc97ed561fe7e0ccfd826584`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: CA-1M: CC BY-NC-ND 4.0; RGB frames from ARKitScenes under the ARKitScenes license.

- `tasks/B250_154/media/photo5.jpg` — dataset photo, 1920×1440, 0.4 MB  
  Source: decode index 2779 of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/43828368/43828368.mov>  
  Cut: CA-1M scene 43828368, frame ca_92155031701583; processing: rot90, jpeg, jpeg  
  sha256 `9d275031fc89f41d7c7b98ba15a3789a11e5469bd94108e8fefb5a87ea8e4f20`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: CA-1M: CC BY-NC-ND 4.0; RGB frames from ARKitScenes under the ARKitScenes license.

### B250_155

- `tasks/B250_155/media/photo.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/390eda9157/dslr/resized_undistorted_images/6d19dce0_DSC07310.JPG`, source sha256 `8747f61a30e6e1c3c88cc3853eaa8ba6632390dc42af7ddf700795f6cbfe9a2c`  
  Source: member `390eda9157/dslr/resized_undistorted_images/6d19dce0_DSC07310.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0005.zip>, source sha256 `8747f61a30e6e1c3c88cc3853eaa8ba6632390dc42af7ddf700795f6cbfe9a2c`  
  Cut: ScanNet++ scene 390eda9157, frame 6d19dce0_DSC07310; processing: jpeg  
  sha256 `d71a109387fdd5028813a90a89c7a9bbc58a48a6985f3c0ee3b88f9072fd38c3`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_156

- `tasks/B250_156/media/photo.jpg` — dataset photo, 1920×1440, 0.6 MB  
  Source: decode index 689 of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/45260903/45260903.mov>  
  Cut: CA-1M scene 45260903, frame ca_1529976055041; processing: rot90, jpeg, jpeg  
  sha256 `63afc9e69d50ef29a40caba07f9822faab72c58f288be227e0aa2b2f5f3ee9eb`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: CA-1M: CC BY-NC-ND 4.0; RGB frames from ARKitScenes under the ARKitScenes license.

### B250_157

- `tasks/B250_157/media/photo.jpg` — dataset photo, 1920×1440, 0.5 MB  
  Source: decode index 425 of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/47332885/47332885.mov>  
  Cut: CA-1M scene 47332885, frame ca_7491246125875; processing: rot90, jpeg, jpeg  
  sha256 `27ea25ae56727a1cb1f70c1aecbe3eea0ee4bf536dcdd77e0cdb80450ae6ece2`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: CA-1M: CC BY-NC-ND 4.0; RGB frames from ARKitScenes under the ARKitScenes license.

### B250_158

- `tasks/B250_158/media/photo.jpg` — dataset photo, 1920×1440, 0.4 MB  
  Source: decode index 773 of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/48018757/48018757.mov>  
  Cut: CA-1M scene 48018757, frame ca_14865747880833; processing: rot90, jpeg, jpeg  
  sha256 `34a2646bcfd9d8bc17ae87e46d33b038606f97d705e728227094bfc383e62cb3`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: CA-1M: CC BY-NC-ND 4.0; RGB frames from ARKitScenes under the ARKitScenes license.

### B250_159

- `tasks/B250_159/media/photo.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/c50d2d1d42/dslr/resized_images/DSC01620.JPG`, source sha256 `f935924e12293583194055a40048d08df1baefc40a365d1563967617fc3e8d84`  
  Source: member `scannetpp_eval/data/c50d2d1d42/dslr/resized_images/DSC01620.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `f935924e12293583194055a40048d08df1baefc40a365d1563967617fc3e8d84`  
  Cut: ScanNet++ scene c50d2d1d42, frame DSC01620; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `cff57debc0b5f0b3214692980ee83cf32add210576371da5d3e8f2f50bcbb0dd`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_160

- `tasks/B250_160/media/photo.jpg` — dataset photo, 1752×1168, 0.5 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/3caf4324fd/dslr/resized_undistorted_images/852b663a_DSC05372.JPG`, source sha256 `448d1ec34ff36971264e0286e73315379f840f9c79b539f734ffd75554ede671`  
  Source: member `3caf4324fd/dslr/resized_undistorted_images/852b663a_DSC05372.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0001.zip>, source sha256 `448d1ec34ff36971264e0286e73315379f840f9c79b539f734ffd75554ede671`  
  Cut: ScanNet++ scene 3caf4324fd, frame 852b663a_DSC05372; processing: jpeg  
  sha256 `63e7b44fca948688a0f1037ea96e11f9399e9230e540a8fc08329eea7ad678c9`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_161

- `tasks/B250_161/media/photo.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/acd95847c5/dslr/resized_images/DSC08698.JPG`, source sha256 `d0b7c58d31b4d85797018051fea5dbc97d5bfd4bd7bcd484c6f937e94cace3b9`  
  Source: member `scannetpp_eval/data/acd95847c5/dslr/resized_images/DSC08698.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `d0b7c58d31b4d85797018051fea5dbc97d5bfd4bd7bcd484c6f937e94cace3b9`  
  Cut: ScanNet++ scene acd95847c5, frame DSC08698; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `be411db51f2cf11c0ad127fc452724119e565958fa3e4f346f487c9a0c8aea82`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_162

- `tasks/B250_162/media/photo.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/6115eddb86/dslr/resized_images/DSC09325.JPG`, source sha256 `a2f07418fb6d99935f5d1fb1ab932c2b3fa63a9a92ce53569b12eeb61cf52941`  
  Source: member `scannetpp_eval/data/6115eddb86/dslr/resized_images/DSC09325.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `a2f07418fb6d99935f5d1fb1ab932c2b3fa63a9a92ce53569b12eeb61cf52941`  
  Cut: ScanNet++ scene 6115eddb86, frame DSC09325; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `727f6f619aaed4658ee63b26503e7d7560990f67c8c569a10e14f9dbd5d88518`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_163

- `tasks/B250_163/media/photo.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/5eb31827b7/dslr/resized_images/DSC02938.JPG`, source sha256 `50ac7322e268237f8a9cc5453e9cc649d198fa04db368c58661e13e8844be2b1`  
  Source: member `scannetpp_eval/data/5eb31827b7/dslr/resized_images/DSC02938.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `50ac7322e268237f8a9cc5453e9cc649d198fa04db368c58661e13e8844be2b1`  
  Cut: ScanNet++ scene 5eb31827b7, frame DSC02938; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `b69d458976de744ad6b4caa48601efe4eee0e076f73dad64c70143400439960b`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_164

- `tasks/B250_164/media/photo.jpg` — dataset photo, 1752×1168, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/7831862f02/dslr/resized_images/DSC04186.JPG`, source sha256 `a4751a19e8657e19c74cab3b8c1c0b3781d6fca910530c1b7c22e24fcbc17f09`  
  Source: member `scannetpp_eval/data/7831862f02/dslr/resized_images/DSC04186.JPG` of <https://huggingface.co/datasets/kairunwen/scannetpp/resolve/6b6adcabc7395e3ff03fa034f4ffb3068fd03937/scannetpp_eval.zip>, source sha256 `a4751a19e8657e19c74cab3b8c1c0b3781d6fca910530c1b7c22e24fcbc17f09`  
  Cut: ScanNet++ scene 7831862f02, frame DSC04186; processing: fisheye_undistort, jpeg, jpeg  
  sha256 `105cb330afa26cd0cb405a31b8308cc5b2f2571bd9932b9e0775ed45dedcca0a`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_165

- `tasks/B250_165/media/photo.jpg` — dataset photo, 1920×1080, 0.5 MB  
  Official data (optional, used first when you pass `--datasets-root`): `pandaset/051/camera/front_camera/15.jpg`, source sha256 `4c790423fc135d2c241e74166ffc96f976bc70bb5b088a7e697f9c087a7dda3c`  
  Source: member `pandaset/051/camera/front_camera/15.jpg` of <https://huggingface.co/datasets/georghess/pandaset/resolve/e2e123aea3b3132c67f4b395ec6120f63e190271/pandaset.zip>, source sha256 `4c790423fc135d2c241e74166ffc96f976bc70bb5b088a7e697f9c087a7dda3c`  
  Cut: PandaSet scene 051, frame front_camera_15; processing: jpeg  
  sha256 `d6999149c5766eea9be21821fbbc7ff7ec8aa074e1ed8e31b1aa8ed026140dcf`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: PandaSet Terms of Use (CC BY 4.0 with additional terms).

### B250_167

- `tasks/B250_167/media/photo.jpg` — dataset photo, 1752×1168, 0.3 MB  
  Official data (optional, used first when you pass `--datasets-root`): `scannetpp/data/09d6e808b4/dslr/resized_undistorted_images/DSC06092.JPG`, source sha256 `9fcfa007e57adcb21d696476dc865e4a137a72b460624e0cd67bdc72515d8ccc`  
  Source: member `09d6e808b4/dslr/resized_undistorted_images/DSC06092.JPG` of <https://huggingface.co/datasets/yifei-liu/scannetppv2/resolve/c7acf9baf8c4d7cae67d71691cf61a4506fad8d6/scannetppv2_dslr_batch_0003.zip>, source sha256 `9fcfa007e57adcb21d696476dc865e4a137a72b460624e0cd67bdc72515d8ccc`  
  Cut: ScanNet++ scene 09d6e808b4, frame DSC06092; processing: jpeg  
  sha256 `07b3b48ecb12fb8203acbf70f28b41485fdec7b2905462f48840860581692ce7`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ScanNet++ Terms of Use (non-commercial research and education); obtain the data from the official site.

### B250_168

- `tasks/B250_168/media/photo.jpg` — dataset photo, 2074×1382, 0.8 MB  
  Official data (optional, used first when you pass `--datasets-root`): `eth3d/playground/images/dslr_images_undistorted/DSC_0598.JPG`, source sha256 `d73b50480d49ed8e390602c25ab0c0c981dccbac9b8c14fafdfaef155a4d8e59`  
  Source: <https://huggingface.co/datasets/PeterDAI/eth3d/resolve/fb2ddc77390926c335c48dace97533f519fdd571/playground/images/dslr_images_undistorted/DSC_0598.JPG>, source sha256 `d73b50480d49ed8e390602c25ab0c0c981dccbac9b8c14fafdfaef155a4d8e59`  
  Cut: ETH3D scene playground, frame DSC_0598; processing: resize, jpeg, jpeg  
  sha256 `8ef31bc2cd7d2fa3d6859337b4c62ea94a1e9957f1a964be4c9cb24ee37ff900`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ETH3D: CC BY-NC-SA 4.0 (Schops et al., CVPR 2017).

### B250_169

- `tasks/B250_169/media/photo.jpg` — dataset photo, 1920×1080, 0.4 MB  
  Official data (optional, used first when you pass `--datasets-root`): `pandaset/005/camera/left_camera/60.jpg`, source sha256 `6c3017f6bebff61b5c0e6b9b94e0387844e5d2648f7a31387ccd82c5f38b91da`  
  Source: member `pandaset/005/camera/left_camera/60.jpg` of <https://huggingface.co/datasets/georghess/pandaset/resolve/e2e123aea3b3132c67f4b395ec6120f63e190271/pandaset.zip>, source sha256 `6c3017f6bebff61b5c0e6b9b94e0387844e5d2648f7a31387ccd82c5f38b91da`  
  Cut: PandaSet scene 005, frame left_camera_60; processing: jpeg  
  sha256 `1f273ff27c6b2191aba77d5246c116eb3a414284cbc2385b5e668a1c5f83f9d0`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: PandaSet Terms of Use (CC BY 4.0 with additional terms).

### B250_170

- `tasks/B250_170/media/frame1.jpg` — frame from a video, 1482×834 JPEG, 0.1 MB  
  Source: Internet Archive item "Nothing Sacred" <https://archive.org/details/NothingSacredVideoQualityUpgrade> (public-domain release; the frame is searched within ±600 s)  
  Fallback 1: Internet Archive item "Nothing Sacred" <https://archive.org/details/NothingSacred> (public-domain release; the frame is searched within ±600 s)  
  Fallback 2: Internet Archive item "Nothing Sacred" <https://archive.org/details/nothing-sacred-1937_202402> (public-domain release; the frame is searched within ±600 s)  
  Cut: the frame shown at 3876.084 s of the original, crop [0.001, 0.122, 0.997, 0.894] of the frame  
  sha256 `1458dac57b1c57fe542923c02821aa8437c6446786356dd3facbeb1cbcfc1e30`; checked by content fingerprints (`ok` / `aligned`): another release of the same film  
  Terms: Film in the US public domain; Internet Archive releases marked Public Domain.

### B250_172

- `tasks/B250_172/media/frame1.jpg` — frame from a video, 1424×800 JPEG, 0.1 MB  
  Source: Internet Archive item "Royal Wedding" <https://archive.org/details/royal-wedding-1951> (public-domain release; the frame is searched within ±600 s)  
  Fallback 1: Internet Archive item "Royal Wedding" <https://archive.org/details/royal_wedding> (public-domain release; the frame is searched within ±600 s)  
  Cut: the frame shown at 265.635 s of the original, crop [0.000, 0.127, 1.000, 0.873] of the frame  
  sha256 `ad87070112734e7baba65626da5db864f0cb8b48b7729afb9d13c5aec3f64081`; checked by content fingerprints (`ok` / `aligned`): another release of the same film  
  Terms: Film in the US public domain; Internet Archive releases marked Public Domain.

### B250_174

- `tasks/B250_174/media/frame1.jpg` — frame from a video, 1920×1080 JPEG, 0.3 MB  
  Source: NASA "Artemis II ISVV-1 Emergency Egress Demonstration (Day 4 - Pad 39B)" <https://images-assets.nasa.gov/video/KSC-20240812-MH-CSH01_0001-Artemis_II_ISVV-1_Emergency_Egress_Demonstration_Day_4-M8390/KSC-20240812-MH-CSH01_0001-Artemis_II_ISVV-1_Emergency_Egress_Demonstration_Day_4-M8390~orig.mp4> (page: <https://images.nasa.gov/details/KSC-20240812-MH-CSH01_0001-Artemis_II_ISVV-1_Emergency_Egress_Demonstration_Day_4-M8390>)  
  Cut: the frame shown at 276.500 s of the original  
  sha256 `78413d382175dfb430d0fd341f0b150413004034a0c89f1906d75cda1d02c0bc`; checked by sha256, then content fingerprints  
  Terms: NASA media: US Government work, generally not subject to US copyright (NASA media usage guidelines).

### B250_175

- `tasks/B250_175/media/frame1.jpg` — frame from a video, 1708×960 JPEG, 0.2 MB  
  Source: Bilibili "Student assignment: a three-minute action film" <https://www.bilibili.com/video/BV1m1421R7kP>  
  Cut: the frame shown at 64.750 s of the original, crop [0.056, 0.111, 0.946, 1.000] of the frame  
  sha256 `d83f3b373184149f36fb1c68ff605e4319ced5f11beca38b8d94971928c16e06`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_176

- `tasks/B250_176/media/frame1.jpg` — frame from a video, 1424×800 JPEG, 84 KB  
  Source: Wikimedia Commons "SCP: SITE 22 - Short Film (MrKlay)" <https://upload.wikimedia.org/wikipedia/commons/f/f2/SCP-_SITE_22_-_Short_Film.webm> (page: <https://commons.wikimedia.org/wiki/File:SCP-_SITE_22_-_Short_Film.webm>)  
  Cut: the frame shown at 38.500 s of the original, crop [0.129, 0.130, 0.871, 0.870] of the frame  
  sha256 `90d0a4c1b2ee98073ba6223486bb555ef81e907144dcfb9e33b5c6eeb56c49e0`; checked by sha256, then content fingerprints  
  Terms: CC BY 3.0 (see the Commons file page).

### B250_178

- `tasks/B250_178/media/frame1.jpg` — frame from a video, 1920×1080 JPEG, 0.1 MB  
  Source: Bilibili "Japanese one-take ad that works both forwards and in reverse" <https://www.bilibili.com/video/av114968372452072>  
  Cut: the frame shown at 33.500 s of the original  
  sha256 `1427f674332ab856cf4eafb473a1bafc018c5811d7c12f69537197c5ff8f30bd`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_179

- `tasks/B250_179/media/frame1.jpg` — frame from a video, 1920×1080 JPEG, 0.1 MB  
  Source: YouTube "ERASE Bullying (public service announcement)" <https://www.youtube.com/watch?v=c1WBzMLgixM>  
  Cut: the frame shown at 7.000 s of the original  
  sha256 `a294836ec61599871a09cd575f92855bb8cd26b1b6b7c06fbf790d1c3d2a2bba`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_180

- `tasks/B250_180/media/frame1.jpg` — frame from a video, 1216×684 JPEG, 47 KB  
  Source: Bilibili "[HyperX TV commercial, basketball] Confidence unlocks potential" <https://www.bilibili.com/video/av61684184>  
  Cut: the frame shown at 11.000 s of the original, crop [0.183, 0.122, 0.817, 0.756] of the frame  
  sha256 `ec4bd85e5fc7f7dadf112a859af10b8b5cd87b2e5c73666e3824d501d98988cc`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_181

- `tasks/B250_181/media/frame1.jpg` — frame from a video, 1280×720 JPEG, 96 KB  
  Source: Bilibili "Pros show how to ride the SMP skatepark in Shanghai" <https://www.bilibili.com/video/av55387466>  
  Cut: the frame shown at 78.700 s of the original  
  sha256 `5770fd1c9b9cb8d4269f9f5e831cbb264da927ee4eec9429581b68c1e2da9c83`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_182

- `tasks/B250_182/media/frame1.jpg` — frame from a video, 1280×720 JPEG, 43 KB  
  Source: YouTube "Emotional Subaru Outback Ad: Walk Away | Award-Winning Safety**" <https://www.youtube.com/watch?v=NeaW6jdqWGw>  
  Cut: the frame shown at 8.000 s of the original  
  sha256 `7e70c42cc03accad3275603fd10d9b3cb10df7ea83ceb86154641fdb5b4b85b5`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_183

- `tasks/B250_183/media/frame1.jpg` — frame from a video, 1920×1080 JPEG, 0.2 MB  
  Source: YouTube ""Together, we can find a child" - Wireless AMBER Alerts PSA" <https://www.youtube.com/watch?v=NAQUNkEtDpo>  
  Cut: the frame shown at 2.000 s of the original  
  sha256 `c2193d982453a4d82fb6b59843feea771ceab1186e6fe853ee7ec4e97164f481`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_184

- `tasks/B250_184/media/frame1.jpg` — frame from a video, 1920×1078 JPEG, 0.3 MB  
  Source: YouTube "BEACH VOLLEYBALL FIRST PERSON | BEST MOMENTS @Tujh91" <https://www.youtube.com/watch?v=ik5fV4F_OcU>  
  Cut: the frame shown at 597.100 s of the original  
  sha256 `80637e35909583351398731126db2c9288a5729e0aa8d82e006b4407981a7e37`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_185

- `tasks/B250_185/media/frame1.jpg` — frame from a video, 1920×1080 JPEG, 0.2 MB  
  Source: YouTube "Samsung Home Appliances | Connected home appliances you can rely on" <https://www.youtube.com/watch?v=Vt-aXSN7PUA>  
  Cut: the frame shown at 16.000 s of the original  
  sha256 `963ee95d84cd34577188ddd7ed3acf749a2b8c92a2f8296fb43e9351b6cfba34`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_186

- `tasks/B250_186/media/frame1.jpg` — frame from a video, 1920×1080 JPEG, 0.2 MB  
  Source: YouTube "AMERICAN NINJA WARRIOR JUNIOR | Epic Girl Power Racing!" <https://www.youtube.com/watch?v=kxW9rNLoW-8>  
  Cut: the frame shown at 76.500 s of the original  
  sha256 `47d0caa479ad5e20a1692078974405cd9f755fce12e7d0926c0e1f0dc7dd16dd`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_187

- `tasks/B250_187/media/frame1.jpg` — frame from a video, 1664×934 JPEG, 62 KB  
  Source: YouTube "Caitlin Clark From Anywhere | Nike" <https://www.youtube.com/watch?v=p813N-7rGUw>  
  Cut: the frame shown at 6.000 s of the original, crop [0.067, 0.000, 0.933, 1.000] of the frame  
  sha256 `95f4befc56fa06943a5552a551fd7b756c04267879e64be1998a1f08ff9c438a`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_189

- `tasks/B250_189/media/reference.mp4` — video excerpt, 45.0 s, 1090×612, 29.8 fps, no audio, 14.4 MB  
  Source: Bilibili "2016 Crufts dog agility contest: Agility Championship Final, Crufts 2016 (720p)" <https://www.bilibili.com/video/av11209573>  
  Cut: 1427.700–1472.700 s of the original, crop [0.075, 0.000, 0.927, 0.850] of the frame  
  sha256 `71bff723e265623adf140b99dfa35805013e6049fb83f84aa017e6157d11cd78`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_190

- `tasks/B250_190/media/reference.mp4` — video excerpt, 45.0 s, 1728×972, 30 fps, no audio, 54.8 MB  
  Source: YouTube "Incredible Dog Challenge 2025: Freestyle Flying Disc Competition Eastern Regional | NBC Sports" <https://www.youtube.com/watch?v=xEkws7jQt2k>  
  Cut: 142.700–187.700 s of the original, crop [0.050, 0.100, 0.950, 1.000] of the frame  
  sha256 `8ad03c25d825c05f0c7034b0874f42044a8e4e0090a0cba7abfde6df3026450b`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_191

- `tasks/B250_191/media/reference.mp4` — video excerpt, 45.0 s, 1050×592, 25 fps, no audio, 15.9 MB  
  Source: Bilibili "2023 Rolex Grand Prix show-jumping final, jump-off: Rolex Grand Prix CHI Genève 2023 - JumpOff" <https://www.bilibili.com/video/av114281244789133>  
  Cut: 491.200–536.200 s of the original, crop [0.000, 0.136, 0.820, 0.958] of the frame  
  sha256 `4ee758352f213aa1c57b55f5e35ce0e6fe5fe394f39bddac6f3b8c4cc44d12e1`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_192

- `tasks/B250_192/media/reference.mp4` — video excerpt, 45.0 s, 1708×960, 24 fps, no audio, 20.9 MB  
  Source: YouTube "The Farmer's Dog: Forever 2023 Super Bowl Commercial" <https://www.youtube.com/watch?v=IAIo-pUDl0s>  
  Cut: 4.000–49.000 s of the original, crop [0.056, 0.056, 0.946, 0.944] of the frame  
  sha256 `2f85ba4c3c6f92a8250374bdc25c475ae0956b22e4a3942dda62373c94cb2911`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_193

- `tasks/B250_193/media/reference.mp4` — video excerpt, 42.0 s, 1280×720, 30 fps, no audio, 14.2 MB  
  Source: YouTube <https://www.youtube.com/watch?v=ENUiRppSjVs>  
  Cut: 648.300–690.300 s of the original  
  sha256 `f3ad6a7a815d8336d4f03ed92373d47fe477c78b7ed15906328bdc3447dad856`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_194

- `tasks/B250_194/media/reference.mp4` — video excerpt, 50.0 s, 1280×720, 30 fps, no audio, 21.6 MB  
  Source: YouTube <https://www.youtube.com/watch?v=9DVvxUaM1YI>  
  Cut: 260.000–310.000 s of the original  
  sha256 `1fa7ea969f7905f098f03440289d9dcd5d8f5485a9a9d4088dd8474f1c00da4f`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_195

- `tasks/B250_195/media/reference.mp4` — video excerpt, 25.0 s, 1280×720, 30 fps, no audio, 14.1 MB  
  Source: YouTube <https://www.youtube.com/watch?v=widUSDYKOPk>  
  Cut: 0.000–25.000 s of the original  
  sha256 `d815c1f7f5401e22e94f0b9160482235f6d8147ba42400d522d56c105c4c842c`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_196

- `tasks/B250_196/media/reference.mp4` — video excerpt, 33.0 s, 1280×720, 24 fps, no audio, 12.0 MB  
  Source: Eyepetizer (kaiyanapp) <https://www.eyepetizer.net/detail.html?vid=177109>  
  Cut: 7.000–40.000 s of the original  
  sha256 `d0fa86869db282e3e5042b35a33f18bc56a4245689078601cfb895254a5b3c61`; checked by sha256, then content fingerprints  
  Terms: Eyepetizer terms of service; copyright remains with the creator.

### B250_197

- `tasks/B250_197/media/reference.mp4` — video excerpt, 25.0 s, 1920×1080, 25 fps, no audio, 24.7 MB  
  Source: Bilibili <https://www.bilibili.com/video/av992468488>  
  Cut: 5.800–30.800 s of the original  
  sha256 `c0d4a45423b52b780132176d5d5429815639e73dff9d5464ce24899371b2e752`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_198

- `tasks/B250_198/media/reference.mp4` — video excerpt, 25.0 s, 1280×720, 24 fps, no audio, 8.7 MB  
  Source: YouTube <https://www.youtube.com/watch?v=dfymiSp3VLI>  
  Cut: 70.000–95.000 s of the original  
  sha256 `af3d101b3a5a469b42f839908187bb65b2c470229218ebb5508dae6c123cdc7a`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_199

- `tasks/B250_199/media/reference.mp4` — video excerpt, 42.0 s, 1280×720, 30 fps, no audio, 23.5 MB  
  Source: YouTube <https://www.youtube.com/watch?v=NXrqtIjGgcM>  
  Cut: 585.000–627.000 s of the original  
  sha256 `86d455b253e9bd360ed474b125bede2b843a9a82137f08d5093c0659d278d691`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_202

- `tasks/B250_202/media/reference.mp4` — video excerpt, 28.0 s, 1920×1080, 24 fps, no audio, 19.7 MB  
  Source: YouTube "MALL COP VS PARKOUR" <https://www.youtube.com/watch?v=FEkZ6DBd4v8>  
  Cut: 62.000–90.000 s of the original  
  sha256 `4b0b4108b2fdc45bf63c5dfd4d507f1383ef9f452c1d5fd8e5ae9a08865d3fec`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_203

- `tasks/B250_203/media/reference.mp4` — video excerpt, 34.5 s, 1656×932, 24 fps, no audio, 27.3 MB  
  Source: Bilibili "Student-made martial-arts short film "Pursuit"" <https://www.bilibili.com/video/BV1Xq421c7Si>  
  Cut: 69.500–104.000 s of the original, crop [0.069, 0.081, 0.931, 0.944] of the frame  
  sha256 `7d7c0d01dcf43c8b954d389eb0e0b0d6823e6718ce5c7b83aa7fa75387bfafd2`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_204

- `tasks/B250_204/media/reference.mp4` — video excerpt, 37.0 s, 1920×1080, 30 fps, no audio, 17.9 MB  
  Source: Bilibili "Chase-scene assignment" <https://www.bilibili.com/video/BV1ns411x7GD>  
  Cut: 2.000–39.000 s of the original  
  sha256 `5cfdae23a8e501e11b85ec368b5d703a60a0dd82bd1af04c24472e782b06561f`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_205

- `tasks/B250_205/media/reference.mp4` — video excerpt, 46.0 s, 1544×868, 30 fps, no audio, 30.4 MB  
  Source: Bilibili "[Film compilation] Running and chases in films - Episode 2" <https://www.bilibili.com/video/BV1gPyxYGEbE>  
  Cut: 96.000–142.000 s of the original, crop [0.098, 0.126, 0.902, 0.930] of the frame  
  sha256 `a2790a0c7d87d6118443d8e5fddc875e25cf17c6fe489def1082272eb334fd2a`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator. Feature-film scene re-uploaded by a user; film rights belong to the studio.

### B250_206

- `tasks/B250_206/media/reference.mp4` — video excerpt, 44.0 s, 1920×1080, 24 fps, no audio, 44.2 MB  
  Source: Bilibili "Honghuagang police special patrol unit: drill on handling an individual extreme act" <https://www.bilibili.com/video/BV19ZD7BPE5i>  
  Cut: 25.500–69.500 s of the original  
  sha256 `7558cfba190e48b0c608ff27ff706a92ff4fdfa93210957d611e59885e3a10e6`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_207

- `tasks/B250_207/media/reference.mp4` — video excerpt, 44.5 s, 1920×1080, 24 fps, no audio, 51.9 MB  
  Source: NASA "Artemis II ISVV-1 Emergency Egress Demonstration (Day 2)" <https://images-assets.nasa.gov/video/KSC-20240809-MH-GEB01_0001-Artemis_II_ISVV-1_Emergency_Egress_Day_2-M8390/KSC-20240809-MH-GEB01_0001-Artemis_II_ISVV-1_Emergency_Egress_Day_2-M8390~orig.mp4> (page: <https://images.nasa.gov/details/KSC-20240809-MH-GEB01_0001-Artemis_II_ISVV-1_Emergency_Egress_Day_2-M8390>)  
  Cut: 107.500–152.000 s of the original  
  sha256 `f0b796f0ba6b0d42ba62d42b42027c9e38d7202c82ba9b1162b47cfce8641344`; checked by sha256, then content fingerprints  
  Terms: NASA media: US Government work, generally not subject to US copyright (NASA media usage guidelines).

### B250_208

- `tasks/B250_208/media/reference.mp4` — video excerpt, 26.0 s, 1708×960, 24 fps, no audio, 20.2 MB  
  Source: YouTube "The Warehouse Fight | REACHER Season 4 | Prime Video" <https://www.youtube.com/watch?v=pR6-kHpY-Tw>  
  Cut: 43.000–69.000 s of the original, crop [0.056, 0.056, 0.946, 0.944] of the frame  
  sha256 `80eed87e36698ccc56a747e2535987f36a2adc95edc9394598bab67de497cd05`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_209

- `tasks/B250_209/media/reference.mp4` — video excerpt, 46.0 s, 1920×1080, 24 fps, no audio, 29.7 MB  
  Source: Bilibili "Central Academy of Drama, film directing class of 2018, first-year short film "Tailing"" <https://www.bilibili.com/video/BV1ct411a7Pv>  
  Cut: 130.000–176.000 s of the original  
  sha256 `e976da72bc9e8e9620a483108b0568843c62aa888c5ee8eff10c02557d056c9b`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_210

- `tasks/B250_210/media/reference.mp4` — video excerpt, 24.1 s, 1920×1080, 24 fps, no audio, 18.2 MB  
  Source: YouTube "SKYFALL | Train Fight – Daniel Craig, Naomie Harris | James Bond" <https://www.youtube.com/watch?v=etnCBggA4Os>  
  Cut: 2.000–26.100 s of the original  
  sha256 `19173d0b87b14f517d62d59661daf6b3437eefeae844bc110092e602f1356830`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_211

- `tasks/B250_211/media/reference.mp4` — video excerpt, 32.0 s, 1920×1080, 25 fps, no audio, 26.6 MB  
  Source: YouTube "ETA – London Parkour Chase Short Action Film!" <https://www.youtube.com/watch?v=h_KxFwsFppI>  
  Cut: 256.000–288.000 s of the original  
  sha256 `079562c1f58ced2dc2a6178a44d15774dcee9e61fca23f70f3fc08cd10370bc3`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_212

- `tasks/B250_212/media/reference.mp4` — video excerpt, 27.0 s, 1536×864, 30 fps, no audio, 20.1 MB  
  Source: Bilibili "[Film compilation] Running and chases in films - Episode 2" <https://www.bilibili.com/video/BV1gPyxYGEbE>  
  Cut: 1736.000–1763.000 s of the original, crop [0.100, 0.130, 0.900, 0.930] of the frame  
  sha256 `bc6d91eb60cf6cf3128b9adbbaa692fa417d5bbaa910361540eafa1d97f97aa2`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator. Feature-film scene re-uploaded by a user; film rights belong to the studio.

### B250_213

- `tasks/B250_213/media/reference.mp4` — video excerpt, 12.9 s, 1464×824, 24 fps, no audio, 6.6 MB  
  Source: Bilibili "Cross-cutting chase-scene assignment by a Korean exchange student (shot entirely on iPhone except the drone shots)" <https://www.bilibili.com/video/BV18o4y1n7dc>  
  Cut: 97.300–110.200 s of the original, crop [0.119, 0.119, 0.881, 0.881] of the frame  
  sha256 `1a7b68447c187e248ee2a798db9c17915ad392afd354263fd9c20da83f8b69fa`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_214

- `tasks/B250_214/media/reference.mp4` — video excerpt, 46.0 s, 1452×816, 24 fps, no audio, 21.1 MB  
  Source: Bilibili "[Action / chase] The Alley (UIC CTV chase-scene assignment)" <https://www.bilibili.com/video/BV1Tb41157ZP>  
  Cut: 78.000–124.000 s of the original, crop [0.123, 0.122, 0.879, 0.878] of the frame  
  sha256 `cf22cea1ad17175ff150eb90535d2f71ed2523ccf45a49aaf3f537e799b1b932`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_215

- `tasks/B250_215/media/reference.mp4` — video excerpt, 48.0 s, 1436×808, 24 fps, no audio, 37.6 MB  
  Source: Bilibili "Mission: Impossible - Fallout: Tom Cruise's motorcycle escape scene" <https://www.bilibili.com/video/BV1e4411v7NT>  
  Cut: 98.000–146.000 s of the original, crop [0.127, 0.103, 0.875, 0.899] of the frame  
  sha256 `791a2a8d2ddc3af9f0d41268f24e2c12842aae49b7111e772caa98a164117d4a`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator. Feature-film scene re-uploaded by a user; film rights belong to the studio.

### B250_216

- `tasks/B250_216/media/reference.mp4` — video excerpt, 14.4 s, 1364×768, 30 fps, no audio, 11.9 MB  
  Source: Bilibili "Cinematography class assignment: chase scene" <https://www.bilibili.com/video/BV1gt4y1K7f8>  
  Cut: 74.800–89.200 s of the original, crop [0.146, 0.144, 0.856, 0.856] of the frame  
  sha256 `4be5e2a9b64f41fe4f4d42d798388014c415bc9edaacb3ad1b07d2f9c628e347`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_217

- `tasks/B250_217/media/reference.mp4` — video excerpt, 45.0 s, 1920×1080, 24 fps, no audio, 37.5 MB  
  Source: YouTube "Bike vs Parkour Chase in Ivrea Italy!" <https://www.youtube.com/watch?v=AdRauOrIDuQ>  
  Cut: 112.000–157.000 s of the original  
  sha256 `b9f34ba7a8888e3bdaff3738ab55c3f5f92ac9099e3a6e8ebdbee0a38abd7709`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_218

- `tasks/B250_218/media/reference.mp4` — video excerpt, 45.0 s, 1920×1080, 24 fps, no audio, 52.5 MB  
  Source: YouTube "Assassin's Creed Unity Meets Parkour in Real Life - 4K!" <https://www.youtube.com/watch?v=S8b1zWOgOKA>  
  Cut: 18.000–63.000 s of the original  
  sha256 `15b83d825d784e654c9952898d9579b7aa9f66953c7177c875042702cd4da7d7`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_219

- `tasks/B250_219/media/reference.mp4` — video excerpt, 14.0 s, 1792×1008, 24 fps, no audio, 13.0 MB  
  Source: YouTube "Bike vs Parkour Race through Edinburgh Scotland! 8K!" <https://www.youtube.com/watch?v=qIKzGESwsNk>  
  Cut: 108.000–122.000 s of the original, crop [0.033, 0.000, 0.967, 0.996] of the frame  
  sha256 `4987c0174708af15fd5529406ec368d6e6dc5cadc2db548589345f93bb0f2647`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_220

- `tasks/B250_220/media/reference.mp4` — video excerpt, 46.0 s, 1424×800, 30 fps, no audio, 19.0 MB  
  Source: Bilibili ""Gangster Hunter" car-chase scene directed by TG, full version" <https://www.bilibili.com/video/BV1Qt421V7S7>  
  Cut: 176.000–222.000 s of the original, crop [0.129, 0.130, 0.871, 0.870] of the frame  
  sha256 `69bcefbd6c37cfabfa32a0e0f5eebf91bf6a1b2009ee888d4d6fc85cf102b719`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator. Feature-film scene re-uploaded by a user; film rights belong to the studio.

### B250_221

- `tasks/B250_221/media/reference.mp4` — video excerpt, 28.0 s, 1920×1080, 30 fps, no audio, 26.5 MB  
  Source: YouTube "Fabio Wibmer - Fabiolous Escape 2" <https://www.youtube.com/watch?v=1CR0QmCaMTs>  
  Cut: 203.000–231.000 s of the original  
  sha256 `76bd95e36a0d9e67751384b766515eea2688e36bcacf02265680c2fd47daa0b4`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_222

- `tasks/B250_222/media/reference.mp4` — video excerpt, 34.0 s, 1452×816, 24 fps, no audio, 24.7 MB  
  Source: YouTube "FOOT CHASE - A Short Action Scene" <https://www.youtube.com/watch?v=4RNFYtyJQ_Q>  
  Cut: 39.000–73.000 s of the original, crop [0.123, 0.000, 0.879, 1.000] of the frame  
  sha256 `235ffacb448f668842f72986ca667b15c58b76716d1ac1c265ad6a8c8fb63582`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_223

- `tasks/B250_223/media/reference.mp4` — video excerpt, 47.0 s, 1452×816, 24 fps, no audio, 33.1 MB  
  Source: YouTube "The Call | A parkour Chase Film - Team Treast" <https://www.youtube.com/watch?v=h-sENkeaW-w>  
  Cut: 352.000–399.000 s of the original, crop [0.123, 0.095, 0.879, 0.901] of the frame  
  sha256 `976b7c4b72645446d7bedf18c6c706c9c24e4a29709ba115ba2d0875cf292ec0`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_224

- `tasks/B250_224/media/reference.mp4` — video excerpt, 19.0 s, 1536×864, 24 fps, no audio, 16.9 MB  
  Source: YouTube "Parkour vs Helicopter vs Jet Ski - Race Through NYC!" <https://www.youtube.com/watch?v=jHghrMuM7vg>  
  Cut: 168.000–187.000 s of the original, crop [0.100, 0.100, 0.900, 0.900] of the frame  
  sha256 `1e8b9fe52d92f0e264d54d2795f8c814eb87e6b323ef8b760422df18c72eeedc`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_225

- `tasks/B250_225/media/reference.mp4` — video excerpt, 31.0 s, 1620×912, 25 fps, no audio, 32.9 MB  
  Source: YouTube "Pool Hall Brawl (Stunt City Previs Rehearsal)" <https://www.youtube.com/watch?v=DiHd_zGBdM4>  
  Cut: 2.000–33.000 s of the original, crop [0.079, 0.078, 0.923, 0.922] of the frame  
  sha256 `b5fc1e8614666e0d991c71b4fa358b559c09403977be8c1ff815a95822784850`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_226

- `tasks/B250_226/media/reference.mp4` — video excerpt, 40.0 s, 1920×1080, 24 fps, no audio, 42.4 MB  
  Source: YouTube "Prince of Persia Meets PARKOUR in REAL LIFE!" <https://www.youtube.com/watch?v=AhbtJwR8Vno>  
  Cut: 12.000–52.000 s of the original  
  sha256 `a07da1ab69d4e7d901a3ebbd2bc192fd53eb69c7aad42408a996d8a73395d532`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_227

- `tasks/B250_227/media/reference.mp4` — video excerpt, 33.0 s, 1920×1080, 24 fps, no audio, 35.5 MB  
  Source: YouTube "Aladdin Meets Parkour in Real Life - [Uzbekistan 2020]" <https://www.youtube.com/watch?v=JPQNm2d3QIQ>  
  Cut: 53.000–86.000 s of the original  
  sha256 `95ef74d45557e854301bab20bdf57873caf1179d11e10077c47bb4902a5db7b9`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_228

- `tasks/B250_228/media/reference.mp4` — video excerpt, 29.0 s, 1920×1080, 30 fps, no audio, 22.9 MB  
  Source: YouTube "Jason Paul Goes Back in Time" <https://www.youtube.com/watch?v=hpTEzp-6CkM>  
  Cut: 192.000–221.000 s of the original  
  sha256 `650b88231951ccd0f4e0cb4032918dd0735085a49c26f2f91ad9d43fac80cfa6`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_229

- `tasks/B250_229/media/reference.mp4` — video excerpt, 45.0 s, 1436×808, 24 fps, no audio, 31.4 MB  
  Source: YouTube "No Time To Die | Another Classic Bond Chase Scene | Extended Preview" <https://www.youtube.com/watch?v=oKIkvEjI2WE>  
  Cut: 8.000–53.000 s of the original, crop [0.127, 0.126, 0.875, 0.874] of the frame  
  sha256 `dd81e50185a9bf6943e3a44e61a7e74d2cc73201d441d9b6f93899169510c14b`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_230

- `tasks/B250_230/media/reference.mp4` — video excerpt, 45.0 s, 1078×604, 25 fps, no audio, 8.2 MB  
  Source: Bilibili "This one-minute home-furnishing ad sets the bar for seamless editing" <https://www.bilibili.com/video/av814012547>  
  Cut: 2.000–47.000 s of the original, crop [0.078, 0.000, 0.920, 0.839] of the frame  
  sha256 `07f69cdb22585d8e91dacd5d95970faf6755cf09d601a883fb8b4c8527d19af2`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_231

- `tasks/B250_231/media/reference.mp4` — video excerpt, 12.0 s, 1920×1080, 25 fps, no audio, 6.4 MB  
  Source: Bilibili "Oriental Leaf Spring Festival ad "Let's drink tea together for the New Year"" <https://www.bilibili.com/video/av1950095394>  
  Cut: 5.000–17.000 s of the original  
  sha256 `96392839626896ddeee26e90670fe6f0f57327583ff5bd59979617a535fd55e9`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_232

- `tasks/B250_232/media/reference.mp4` — video excerpt, 12.0 s, 1912×1076, 30 fps, no audio, 11.0 MB  
  Source: Bilibili "Jacquemus' superb one-take commercial" <https://www.bilibili.com/video/av370761512>  
  Cut: 10.000–22.000 s of the original, crop [0.002, 0.000, 0.998, 0.996] of the frame  
  sha256 `9708d789bb2ae9a42b79da7ac32280f01cf9754b037ad08b56d0f5a1ca52dcba`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_233

- `tasks/B250_233/media/reference.mp4` — video excerpt, 45.0 s, 1920×1080, 25 fps, no audio, 17.2 MB  
  Source: YouTube "Feel At Home | A Furlenco Original" <https://www.youtube.com/watch?v=aXvaMgNHlVc>  
  Cut: 22.000–67.000 s of the original  
  sha256 `4e91b32d6b2e751922bd98aeed7831925a030d5a9f2bdf329637b64b20214415`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_234

- `tasks/B250_234/media/reference.mp4` — video excerpt, 45.0 s, 952×536, 25 fps, no audio, 7.8 MB  
  Source: YouTube "Best Christmas Commercial Ever" <https://www.youtube.com/watch?v=Z_R4q70_rJg>  
  Cut: 122.200–167.200 s of the original, crop [0.128, 0.125, 0.872, 0.869] of the frame  
  sha256 `6974a6466c102b18b3a44e129bf76bdd95028cc0a603ccac9a4edeebfbfb9076`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_235

- `tasks/B250_235/media/reference.mp4` — video excerpt, 36.0 s, 1078×604, 25 fps, no audio, 9.2 MB  
  Source: Bilibili "CGB credit card x Yin Zheng comedy ad "At the office"" <https://www.bilibili.com/video/av245339880>  
  Cut: 4.000–40.000 s of the original, crop [0.078, 0.000, 0.920, 0.839] of the frame  
  sha256 `81df41e7aa1afc22f74c66479ab5f0e13ee7978c535c4235c167418298befb9a`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_236

- `tasks/B250_236/media/reference.mp4` — video excerpt, 45.0 s, 1920×1080, 30 fps, no audio, 42.4 MB  
  Source: Bilibili "A creative one-take birthday greeting from a company to its boss" <https://www.bilibili.com/video/av61696064>  
  Cut: 46.000–91.000 s of the original  
  sha256 `4fbb7cda32a8e0fe521779d1f39fb15f347c4b6f7b533d06d8453ee2d6866a52`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_237

- `tasks/B250_237/media/reference.mp4` — video excerpt, 45.0 s, 1920×1080, 24 fps, no audio, 28.7 MB  
  Source: Bilibili "[The Office] The best cold open of the US Office" <https://www.bilibili.com/video/av837579015>  
  Cut: 64.000–109.000 s of the original  
  sha256 `7abfbd71ab56ba1e2c0cc109a2430bb64e5429a20db5389b09948af6d9e9d67f`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_238

- `tasks/B250_238/media/reference.mp4` — video excerpt, 45.0 s, 1920×1080, 24 fps, no audio, 31.8 MB  
  Source: YouTube "Parkour PARKOUR - The Office US" <https://www.youtube.com/watch?v=0Kvw2BPKjz0>  
  Cut: 2.000–47.000 s of the original  
  sha256 `f673d8ea5e46efa0c4af221191401ee474db37b5664e5d5f89f491ba9d3830ce`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_239

- `tasks/B250_239/media/reference.mp4` — video excerpt, 12.0 s, 1692×952, 30 fps, no audio, 7.1 MB  
  Source: Bilibili "GoPro Awards: indoor skatepark FPV | Bunkeberget" <https://www.bilibili.com/video/av632282040>  
  Cut: 23.000–35.000 s of the original, crop [0.060, 0.119, 0.942, 1.000] of the frame  
  sha256 `d1ec5bf900efefb5a4e8a0199de677d63bf6a8d04e7570d26507fe8df6f41db0`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_240

- `tasks/B250_240/media/reference.mp4` — video excerpt, 35.0 s, 1920×1080, 25 fps, no audio, 24.7 MB  
  Source: YouTube "Who will you Share A Coke with?" <https://www.youtube.com/watch?v=0Ab-F7Y3IXc>  
  Cut: 8.000–43.000 s of the original  
  sha256 `07c177e80027e9361a2624b4f355da337fd71babf41fdf0f9e7b1e0683076466`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_241

- `tasks/B250_241/media/reference.mp4` — video excerpt, 45.0 s, 1200×674, 30 fps, no audio, 10.5 MB  
  Source: Bilibili "Imagine | One take, all walks of life" <https://www.bilibili.com/video/av333011767>  
  Cut: 122.000–167.000 s of the original, crop [0.025, 0.047, 0.963, 0.983] of the frame  
  sha256 `22d9314284cd4f464c9b6d6784838321cbc32a8a85e8a8a6d8186b6deab6b996`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_243

- `tasks/B250_243/media/reference.mp4` — video excerpt, 33.0 s, 1592×896, 24 fps, no audio, 19.0 MB  
  Source: Bilibili "Japanese one-take ad that took 49 takes to get right" <https://www.bilibili.com/video/av527271723>  
  Cut: 55.400–88.400 s of the original, crop [0.085, 0.000, 0.915, 0.830] of the frame  
  sha256 `6f00c3c6516d211d760bbffce128070619f7f206f7e85b74244caa61cbec5a09`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_244

- `tasks/B250_244/media/reference.mp4` — video excerpt, 12.0 s, 1920×1080, 24 fps, no audio, 8.0 MB  
  Source: YouTube "Coca-Cola // Share A Coke" <https://www.youtube.com/watch?v=YjJkkZ6cT1w>  
  Cut: 2.000–14.000 s of the original  
  sha256 `88155ce56e9fdebb737b6559510191bf037fd25f0dbf763364a50e8b6e46e69a`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_245

- `tasks/B250_245/media/reference.mp4` — video excerpt, 45.0 s, 1920×1080, 25 fps, no audio, 46.1 MB  
  Source: YouTube "EPIC FPV Drone One Shot in the Bar" <https://www.youtube.com/watch?v=tC7cDmbqko8>  
  Cut: 30.000–75.000 s of the original  
  sha256 `8bfe324f48747fbe0f96f183c433fc5299a4746e577337ae9d3078eebded8ead`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_246

- `tasks/B250_246/media/reference.mp4` — video excerpt, 45.0 s, 1920×1080, 30 fps, no audio, 55.5 MB  
  Source: Bilibili "[Storm] First-person rooftop parkour: Brighton Rooftop Parkour POV" <https://www.bilibili.com/video/av25423580>  
  Cut: 74.800–119.800 s of the original  
  sha256 `0165948070780f1f72f9044d171a449666080bd2240a998aa22bb44cb1cfe491`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_247

- `tasks/B250_247/media/reference.mp4` — video excerpt, 34.0 s, 1636×920, 25 fps, no audio, 35.9 MB  
  Source: Bilibili "[Re-upload with Chinese subtitles] WDSC (World Downhill Skateboarding Championship) 2026, round 2: Isle of Man" <https://www.bilibili.com/video/av117432106615287>  
  Cut: 455.300–489.300 s of the original, crop [0.075, 0.148, 0.927, 1.000] of the frame  
  sha256 `9a8caf585ff5e5ac7a96cedf1f5ef3c6b5a6bc09f514ab8c6f660541191a174c`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_248

- `tasks/B250_248/media/reference.mp4` — video excerpt, 35.0 s, 1920×1080, 25 fps, no audio, 35.6 MB  
  Source: Bilibili "Proper Perfection! Mercedes double pit stop, 2019 Chinese Grand Prix in Shanghai, the 1,000th F1 race" <https://www.bilibili.com/video/av997389648>  
  Cut: 4.000–39.000 s of the original  
  sha256 `78174283d5b35d6a0d327462377e2a5c9481033893a229aad57e6bb726604ede`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_250

- `tasks/B250_250/media/reference.mp4` — video excerpt, 45.0 s, 1616×908, 30 fps, no audio, 38.2 MB  
  Source: YouTube "Average Hoopers VS D1 Basketball Team" <https://www.youtube.com/watch?v=sgR2wY6Cgvk>  
  Cut: 367.500–412.500 s of the original, crop [0.079, 0.000, 0.921, 0.841] of the frame  
  sha256 `d502c8fe3255b5c0fbeb7a5fc136a22f4df782be01a6cbdc00e83f3e0b53a4ab`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_251

- `tasks/B250_251/media/reference.mp4` — video excerpt, 12.0 s, 1920×1080, 24 fps, no audio, 10.7 MB  
  Source: YouTube "Parkour Kids Training At Sessions Gym" <https://www.youtube.com/watch?v=u9aHufCJ8KY>  
  Cut: 2.000–14.000 s of the original  
  sha256 `87adeadce092d7c7d896bf3b8814af8142c6d3609487bcdfd393e625ecea1bfa`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_252

- `tasks/B250_252/media/reference.mp4` — video excerpt, 12.0 s, 1920×1080, 25 fps, no audio, 13.7 MB  
  Source: YouTube "Lucozade | Bring The Energy | Football | 30" | New TV Ad" <https://www.youtube.com/watch?v=JbGwzQY21ow>  
  Cut: 2.000–14.000 s of the original  
  sha256 `f3e34aaa75ec0cd331db6ced36fb8ec0bae5eaef57f3b293e3dd31c0590fb480`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_253

- `tasks/B250_253/media/reference.mp4` — video excerpt, 45.0 s, 1480×832, 30 fps, no audio, 25.8 MB  
  Source: YouTube "World Championships 2026 - Men’s STREET Semifinals" <https://www.youtube.com/watch?v=p_HkDok3rUA>  
  Cut: 277.100–322.100 s of the original, crop [0.229, 0.115, 1.000, 0.885] of the frame  
  sha256 `bf67e3a73af0145ea6f9a39fa763297d93557b9034837cf419e7a906209df63b`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_254

- `tasks/B250_254/media/reference.mp4` — video excerpt, 45.0 s, 1572×884, 30 fps, no audio, 29.0 MB  
  Source: YouTube "Best of 2025 Masters Agility Championships from Westminster Kennel Club | FOX Sports" <https://www.youtube.com/watch?v=bRV-zcsJhnE>  
  Cut: 14.000–59.000 s of the original, crop [0.092, 0.000, 0.910, 0.819] of the frame  
  sha256 `1c458c00f3cb52698d4a24e08d7b43d72a7f24133be17427a41422fe3c06f0bd`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_255

- `tasks/B250_255/media/reference.mp4` — video excerpt, 35.0 s, 1920×1080, 25 fps, no audio, 37.8 MB  
  Source: YouTube "TOTAL WIPEOUT CHALLENGE vs MY SISTER & BRO" <https://www.youtube.com/watch?v=VopB3HnOJQQ>  
  Cut: 617.500–652.500 s of the original  
  sha256 `a6ddcfc2f474b501801a6e768982133eb7c6ae84fcac2362d413deb0391d5695`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_256

- `tasks/B250_256/media/reference.mp4` — video excerpt, 45.0 s, 1920×1080, 30 fps, no audio, 75.8 MB  
  Source: YouTube "AMERICAN NINJA WARRIOR JUNIOR | Paxton from Ninja Kidz TV Hangs on for the Win!" <https://www.youtube.com/watch?v=OdzW3HoSOvY>  
  Cut: 83.100–128.100 s of the original  
  sha256 `2f9ecb896388ae0281bdb54319740df00c85eaf171e563a009e374324ee6575c`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_257

- `tasks/B250_257/media/reference.mp4` — video excerpt, 45.0 s, 1216×684, 30 fps, no audio, 21.1 MB  
  Source: Bilibili "Wedding opening short film, full version" <https://www.bilibili.com/video/av551926756>  
  Cut: 137.900–182.900 s of the original, crop [0.183, 0.122, 0.817, 0.756] of the frame  
  sha256 `74ebe1123ca10149d7ce76d66c27dbc36817900819add62740165532ee3f7a5d`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_258

- `tasks/B250_258/media/reference.mp4` — video excerpt, 45.0 s, 1436×808, 24 fps, no audio, 28.6 MB  
  Source: YouTube "The Fate of the Furious | Harpooning Dom's Car" <https://www.youtube.com/watch?v=cOdynPv8cok>  
  Cut: 136.700–181.700 s of the original, crop [0.127, 0.126, 0.875, 0.874] of the frame  
  sha256 `86e3fe5f5c0ebdc5aeee782712e766024f322186b39465e9003df3f89fdfcf51`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_259

- `tasks/B250_259/media/reference.mp4` — video excerpt, 45.0 s, 1436×808, 24 fps, no audio, 24.8 MB  
  Source: YouTube "Infinite (2021) - Crazy Car Chase Scene | Movieclips" <https://www.youtube.com/watch?v=7DXP7HsnL2A>  
  Cut: 60.000–105.000 s of the original, crop [0.127, 0.126, 0.875, 0.874] of the frame  
  sha256 `ee8c3f537a70525689e9f56db88512bfad174e2b74fa961e453189f8d7f258bc`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_260

- `tasks/B250_260/media/reference.mp4` — video excerpt, 45.0 s, 1464×824, 24 fps, no audio, 24.5 MB  
  Source: YouTube "FASTER: A Chapter From The Cure (Short film)" <https://www.youtube.com/watch?v=RQDntmtmIbI>  
  Cut: 173.600–218.600 s of the original, crop [0.119, 0.119, 0.881, 0.881] of the frame  
  sha256 `c7defb1e378cf6ba833ce757699a41d72fb4997fd3d9d03d1f4fa619426e2e45`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_261

- `tasks/B250_261/media/reference.mp4` — video excerpt, 35.0 s, 1920×1080, 30 fps, no audio, 59.8 MB  
  Source: YouTube "Paris Rooftop Parkour POV 🇫🇷" <https://www.youtube.com/watch?v=H6AO-RhaG_4>  
  Cut: 22.000–57.000 s of the original  
  sha256 `cf546a398cdade61f8ec50e6e7b2e74acf77dffcfc91d80ec6ae955d146cd714`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### B250_262

- `tasks/B250_262/media/reference.mp4` — video excerpt, 45.0 s, 1616×908, 25 fps, no audio, 27.5 MB  
  Source: Bilibili "The toughest brake-pad commercial ever" <https://www.bilibili.com/video/av1601114521>  
  Cut: 2.000–47.000 s of the original, crop [0.079, 0.000, 0.921, 0.841] of the frame  
  sha256 `aa1fb7502db6ca236e92d80800fa8c321f2ee69b6337c3f96f64909fdec4c2ec`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### B250_264

- `tasks/B250_264/media/reference.mp4` — video excerpt, 30.0 s, 1920×1080, 25 fps, no audio, 46.7 MB  
  Source: YouTube "2026 X-ADV Special Edition | Honda Motorcycles" <https://www.youtube.com/watch?v=Tr1G8g7oiL4>  
  Cut: 11.000–41.000 s of the original  
  sha256 `4237a6a5532c4b7cd087abdcca07f1c1e87b3ad2605eade5bbcd8ded1443559f`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### M50_017

- `tasks/M50_017/media/frame1.jpg` — frame from a video, 1920×1080 JPEG, 0.2 MB  
  Source: YouTube <https://www.youtube.com/watch?v=-3kEBPikrbc>  
  Cut: the frame shown at 901.900 s of the original  
  sha256 `abce4e5bf0a92695a61c5e3f6b3197e4fa3264a820ea15665cc4b8c66bf6feaa`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### M50_018

- `tasks/M50_018/media/frame1.jpg` — frame from a video, 1280×720 JPEG, 0.2 MB  
  Source: YouTube <https://www.youtube.com/watch?v=eAZwx6D7Bu4>  
  Cut: the frame shown at 741.900 s of the original  
  sha256 `741318c0aaa22d101f332b69f549f45011aed54efb3f7a46ce1c474a1829beee`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### M50_019

- `tasks/M50_019/media/frame1.jpg` — frame from a video, 852×480 JPEG, 0.1 MB  
  Source: Bilibili <https://www.bilibili.com/video/av99139608>  
  Cut: the frame shown at 32.333 s of the original  
  sha256 `1357e5cf39775a3f511ee48ee302193b4dce696e0c5ba92bc9fa2cd894cfa397`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### M50_020

- `tasks/M50_020/media/frame1.jpg` — frame from a video, 1280×720 JPEG, 0.1 MB  
  Source: YouTube <https://www.youtube.com/watch?v=u9YpzpYei_U>  
  Cut: the frame shown at 752.033 s of the original  
  sha256 `b798f12c9a4c668dd6fcd9068157d2cedd8e0db4ba77589508756130783baa0f`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### M50_021

- `tasks/M50_021/media/frame1.jpg` — frame from a video, 1280×720 JPEG, 0.1 MB  
  Source: Bilibili <https://www.bilibili.com/video/av992715074>  
  Cut: the frame shown at 365.967 s of the original  
  sha256 `b0e50fa229bd9d900468e05aab6b19584c4c1b6f3f9a4325ce207c42c24e61b3`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### M50_022

- `tasks/M50_022/media/frame1.jpg` — frame from a video, 1280×720 JPEG, 0.1 MB  
  Source: YouTube <https://www.youtube.com/watch?v=_vhG-7fiMzs>  
  Cut: the frame shown at 766.033 s of the original  
  sha256 `8f2001648561344e8102b7083bf4e4bacc0d7793a3564f9596fd1c5841294669`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### M50_023

- `tasks/M50_023/media/frame1.jpg` — frame from a video, 1280×720 JPEG, 0.3 MB  
  Source: YouTube <https://www.youtube.com/watch?v=z0z9FZNLRfw>  
  Cut: the frame shown at 4.033 s of the original  
  sha256 `d0c65f99e046b9f615e6f6fc3af9134b98da7ea45e0db28dc8d683fefdc0be39`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### M50_024

- `tasks/M50_024/media/frame1.jpg` — frame from a video, 1280×720 JPEG, 83 KB  
  Source: Bilibili <https://www.bilibili.com/video/av985957804>  
  Cut: the frame shown at 92.900 s of the original  
  sha256 `825ce2d7b86ecd30fe8c0d07fa48527770677a926458a12371084c7d9d40e0da`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### M50_025

- `tasks/M50_025/media/photo1.jpg` — dataset photo, 99 KB  
  Source: member `*_78058.137.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42446103/vga_wide.zip>  
  Cut: ARKitScenes scene 42446103, frame timestamp 78058.137; processing: rot90, jpeg  
  sha256 `ac20e013b9b241c0026be1235d56a79ce847df1d78beec7e9a4f431ef79d289f`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_025/media/photo2.jpg` — dataset photo, 74 KB  
  Source: member `*_78084.327.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42446103/vga_wide.zip>  
  Cut: ARKitScenes scene 42446103, frame timestamp 78084.327; processing: rot90, jpeg  
  sha256 `9fa307cbfa26944f54b0560a5160ad12ba12ec1ed52eb8f3527693489cc0e886`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_025/media/photo3.jpg` — dataset photo, 98 KB  
  Source: member `*_78116.348.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42446103/vga_wide.zip>  
  Cut: ARKitScenes scene 42446103, frame timestamp 78116.348; processing: rot90, jpeg  
  sha256 `6f200c0bd96bbd6bef093ff9cc063ed99b57a6c9bb285628397bfce5f91835b4`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_025/media/photo4.jpg` — dataset photo, 58 KB  
  Source: member `*_78163.062.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42446103/vga_wide.zip>  
  Cut: ARKitScenes scene 42446103, frame timestamp 78163.062; processing: rot90, jpeg  
  sha256 `473f22a805c17c6541c9ac062b2433a9c0c7860d18819085769eaa376a9c124c`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_025/media/photo5.jpg` — dataset photo, 73 KB  
  Source: member `*_78166.628.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42446103/vga_wide.zip>  
  Cut: ARKitScenes scene 42446103, frame timestamp 78166.628; processing: rot90, jpeg  
  sha256 `028d57588006a0bd0fd1ea55d0f4f3a6d92481f67f1458fbbc2633475cfa270f`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

### M50_026

- `tasks/M50_026/media/photo1.jpg` — dataset photo, 59 KB  
  Source: member `*_201563.273.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42446541/vga_wide.zip>  
  Cut: ARKitScenes scene 42446541, frame timestamp 201563.273; processing: rot90, jpeg  
  sha256 `d855d16ba2604b2c46f6f4d6762a34954c96d7e63fd18c6730828ef1e0925a09`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_026/media/photo2.jpg` — dataset photo, 68 KB  
  Source: member `*_201613.053.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42446541/vga_wide.zip>  
  Cut: ARKitScenes scene 42446541, frame timestamp 201613.053; processing: rot90, jpeg  
  sha256 `7cd013dfb7391097a897aeed86fee70fcfd418871860baf440b259ef824a996e`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_026/media/photo3.jpg` — dataset photo, 94 KB  
  Source: member `*_201627.913.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42446541/vga_wide.zip>  
  Cut: ARKitScenes scene 42446541, frame timestamp 201627.913; processing: rot90, jpeg  
  sha256 `32285f401f9e7dd331d0317f2cfc8ff9e4a6709b4ab359157ee36912086e1e04`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_026/media/photo4.jpg` — dataset photo, 99 KB  
  Source: member `*_201642.907.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42446541/vga_wide.zip>  
  Cut: ARKitScenes scene 42446541, frame timestamp 201642.907; processing: rot90, jpeg  
  sha256 `67dfdddd5bae9b84cd494ad42f20dbda42df6f6fbf1d9acfcf678d8406aaa3d4`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_026/media/photo5.jpg` — dataset photo, 66 KB  
  Source: member `*_201663.465.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42446541/vga_wide.zip>  
  Cut: ARKitScenes scene 42446541, frame timestamp 201663.465; processing: rot90, jpeg  
  sha256 `b75e6839a3bf8eb621f5caa153a0160ff0cc0ee271915005a96ed29428f769da`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

### M50_027

- `tasks/M50_027/media/photo1.jpg` — dataset photo, 62 KB  
  Source: member `*_196464.828.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42898849/vga_wide.zip>  
  Cut: ARKitScenes scene 42898849, frame timestamp 196464.828; processing: rot90, jpeg  
  sha256 `a54f8d1d0b2c20ebe41786db44293be283baf42a33c7bae84c43967338b88c62`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_027/media/photo2.jpg` — dataset photo, 54 KB  
  Source: member `*_196469.459.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42898849/vga_wide.zip>  
  Cut: ARKitScenes scene 42898849, frame timestamp 196469.459; processing: rot90, jpeg  
  sha256 `29931e369c654951fdc6d05fc7ecba02aa760b87dce40a26b0e20023cd85f7e8`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_027/media/photo3.jpg` — dataset photo, 53 KB  
  Source: member `*_196537.099.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42898849/vga_wide.zip>  
  Cut: ARKitScenes scene 42898849, frame timestamp 196537.099; processing: rot90, jpeg  
  sha256 `d1fd4d6f8da22c70b2712c93150cb9cfed1bbef154f404ecd589e59b8471ac41`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_027/media/photo4.jpg` — dataset photo, 63 KB  
  Source: member `*_196541.764.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42898849/vga_wide.zip>  
  Cut: ARKitScenes scene 42898849, frame timestamp 196541.764; processing: rot90, jpeg  
  sha256 `6c0155a19adf33a158e48ac6526d5418531dc071411cccc6c6ee57a346d2b1c5`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_027/media/photo5.jpg` — dataset photo, 57 KB  
  Source: member `*_196577.516.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42898849/vga_wide.zip>  
  Cut: ARKitScenes scene 42898849, frame timestamp 196577.516; processing: rot90, jpeg  
  sha256 `be2da79794594fd049914fca173a698b0041cef7d81c19a0e91f6d3e57f74545`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

### M50_028

- `tasks/M50_028/media/photo1.jpg` — dataset photo, 56 KB  
  Source: member `*_790504.894.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42899685/vga_wide.zip>  
  Cut: ARKitScenes scene 42899685, frame timestamp 790504.894; processing: rot90, jpeg  
  sha256 `6ffc010f619a1783778bfeeaa7a0b12764fe6bb30a7b61af4cddbc63d141dfdb`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_028/media/photo2.jpg` — dataset photo, 66 KB  
  Source: member `*_790518.888.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42899685/vga_wide.zip>  
  Cut: ARKitScenes scene 42899685, frame timestamp 790518.888; processing: rot90, jpeg  
  sha256 `1a88f4a153bf7a174663ae38816f207c265a627409ab5e89e1a7b692344524cb`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_028/media/photo3.jpg` — dataset photo, 73 KB  
  Source: member `*_790545.244.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42899685/vga_wide.zip>  
  Cut: ARKitScenes scene 42899685, frame timestamp 790545.244; processing: rot90, jpeg  
  sha256 `eb53c5f23158fa4c1046c799a4436a53b3e9143027050fd52d8ce3a2fbcc3b66`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_028/media/photo4.jpg` — dataset photo, 0.1 MB  
  Source: member `*_790560.671.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42899685/vga_wide.zip>  
  Cut: ARKitScenes scene 42899685, frame timestamp 790560.671; processing: rot90, jpeg  
  sha256 `485b17a25be9eba65ecf0be7bfcf54bb7f2a4265d6dceab79edd595602c6c13b`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_028/media/photo5.jpg` — dataset photo, 49 KB  
  Source: member `*_790561.737.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/42899685/vga_wide.zip>  
  Cut: ARKitScenes scene 42899685, frame timestamp 790561.737; processing: rot90, jpeg  
  sha256 `a64e4aeb3a6c07868f93017679a49cb6d9117e2ad062464dadfe0c6977a9999e`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

### M50_029

- `tasks/M50_029/media/photo1.jpg` — dataset photo, 89 KB  
  Source: member `*_311470.924.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/47430468/vga_wide.zip>  
  Cut: ARKitScenes scene 47430468, frame timestamp 311470.924; processing: rot90, jpeg  
  sha256 `0e09515c4f522f10adffd7af05acc5c9111370f30301cc989908a6d3fdc2a7ab`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_029/media/photo2.jpg` — dataset photo, 73 KB  
  Source: member `*_311486.617.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/47430468/vga_wide.zip>  
  Cut: ARKitScenes scene 47430468, frame timestamp 311486.617; processing: rot90, jpeg  
  sha256 `ba0e558b4326e6d76b42aae7c0da93b9b73014e75736adf3fd682e7e5713ace9`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_029/media/photo3.jpg` — dataset photo, 76 KB  
  Source: member `*_311489.483.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/47430468/vga_wide.zip>  
  Cut: ARKitScenes scene 47430468, frame timestamp 311489.483; processing: rot90, jpeg  
  sha256 `350b12bba631b04770031a06af03bbbe55bcd2cd92a6bfb3cd4cbbb44fa4fb8d`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_029/media/photo4.jpg` — dataset photo, 83 KB  
  Source: member `*_311505.776.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/47430468/vga_wide.zip>  
  Cut: ARKitScenes scene 47430468, frame timestamp 311505.776; processing: rot90, jpeg  
  sha256 `213941b069925132f1d90822d6bb1fdb8d226317bb75c5d7efd3d759285d27e0`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

- `tasks/M50_029/media/photo5.jpg` — dataset photo, 83 KB  
  Source: member `*_311510.474.png` of <https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/47430468/vga_wide.zip>  
  Cut: ARKitScenes scene 47430468, frame timestamp 311510.474; processing: rot90, jpeg  
  sha256 `15527fb9d7c5d30c484628c0f249f28e981fceba425fd75a06a293b9260fb428`; checked by byte-identical with OpenCV (`identical`); the source file is checked against its sha256 first  
  Terms: ARKitScenes license (Apple, non-commercial research).

### M50_032

- `tasks/M50_032/media/photo1.jpg` — frame from a video, 1920×1080 JPEG, 0.5 MB  
  Source: BMW Group PressClub "BMW x Valeo automated valet parking (press footage PF0009332, HD master)" <https://mediapool.bmwgroup.com/download/edown/tvFootageDownload.mov?dokNo=PF0009332&actEvent=tvFootageMovHd&attachment=1> (page: <https://www.press.bmwgroup.com/global/video/detail/PF0009332/bmwxvaleo-automated-valet-parking>)  
  Cut: the frame shown at 6.150 s of the original  
  sha256 `5520e7c2398a4e10a777153cbeb3c1590e30711f525ed2754cd572a2e2b6ac2a`; checked by sha256, then content fingerprints  
  Terms: BMW Group PressClub press footage under the PressClub terms of use.

- `tasks/M50_032/media/photo2.jpg` — frame from a video, 1920×1080 JPEG, 0.7 MB  
  Source: BMW Group PressClub "BMW x Valeo automated valet parking (press footage PF0009332, HD master)" <https://mediapool.bmwgroup.com/download/edown/tvFootageDownload.mov?dokNo=PF0009332&actEvent=tvFootageMovHd&attachment=1> (page: <https://www.press.bmwgroup.com/global/video/detail/PF0009332/bmwxvaleo-automated-valet-parking>)  
  Cut: the frame shown at 13.017 s of the original  
  sha256 `23ea31e3b9dbdf1f8a085c6dce0b2eac77976c5e174f5b4fe17f7199700296ea`; checked by sha256, then content fingerprints  
  Terms: BMW Group PressClub press footage under the PressClub terms of use.

- `tasks/M50_032/media/photo3.jpg` — frame from a video, 1920×1080 JPEG, 0.6 MB  
  Source: BMW Group PressClub "BMW x Valeo automated valet parking (press footage PF0009332, HD master)" <https://mediapool.bmwgroup.com/download/edown/tvFootageDownload.mov?dokNo=PF0009332&actEvent=tvFootageMovHd&attachment=1> (page: <https://www.press.bmwgroup.com/global/video/detail/PF0009332/bmwxvaleo-automated-valet-parking>)  
  Cut: the frame shown at 23.217 s of the original  
  sha256 `08365ae9ff95549c69e65d399c382b66804aa5e54f22458901d92a720f97d49e`; checked by sha256, then content fingerprints  
  Terms: BMW Group PressClub press footage under the PressClub terms of use.

- `tasks/M50_032/media/photo4.jpg` — frame from a video, 1920×1080 JPEG, 0.6 MB  
  Source: BMW Group PressClub "BMW x Valeo automated valet parking (press footage PF0009332, HD master)" <https://mediapool.bmwgroup.com/download/edown/tvFootageDownload.mov?dokNo=PF0009332&actEvent=tvFootageMovHd&attachment=1> (page: <https://www.press.bmwgroup.com/global/video/detail/PF0009332/bmwxvaleo-automated-valet-parking>)  
  Cut: the frame shown at 44.750 s of the original  
  sha256 `d6186ff942ba1358c4194b0db8975554355c0a146ad5f575095f9c1e71fa915e`; checked by sha256, then content fingerprints  
  Terms: BMW Group PressClub press footage under the PressClub terms of use.

- `tasks/M50_032/media/photo5.jpg` — frame from a video, 1920×1080 JPEG, 0.5 MB  
  Source: BMW Group PressClub "BMW x Valeo automated valet parking (press footage PF0009332, HD master)" <https://mediapool.bmwgroup.com/download/edown/tvFootageDownload.mov?dokNo=PF0009332&actEvent=tvFootageMovHd&attachment=1> (page: <https://www.press.bmwgroup.com/global/video/detail/PF0009332/bmwxvaleo-automated-valet-parking>)  
  Cut: the frame shown at 56.017 s of the original  
  sha256 `99c08f4a267a2da64587dbee5a393bf0d1b4c1f20be1564392bda21b1f13e27d`; checked by sha256, then content fingerprints  
  Terms: BMW Group PressClub press footage under the PressClub terms of use.

### M50_033

- `tasks/M50_033/media/reference.mp4` — video excerpt, 37.9 s, 1280×720, 24 fps, no audio, 13.6 MB  
  Source: Eyepetizer (kaiyanapp) <https://www.eyepetizer.net/detail.html?vid=13041>  
  Cut: 10.050–47.950 s of the original  
  sha256 `6a0878c98790fe8a4b320bb72ae49ecd7e4805426367ff249e8eba8cafb166e0`; checked by sha256, then content fingerprints  
  Terms: Eyepetizer terms of service; copyright remains with the creator.

### M50_034

- `tasks/M50_034/media/reference.mp4` — video excerpt, 43.9 s, 1280×720, 30 fps, no audio, 13.0 MB  
  Source: Eyepetizer (kaiyanapp) <https://www.eyepetizer.net/detail.html?vid=267023>  
  Cut: 4.050–47.950 s of the original  
  sha256 `8875b41b06d74ba1c7c95e8efc879eaddaf02ad5bc9e145547764567d5006ffc`; checked by sha256, then content fingerprints  
  Terms: Eyepetizer terms of service; copyright remains with the creator.

### M50_035

- `tasks/M50_035/media/reference.mp4` — video excerpt, 56.9 s, 1280×532, 24 fps, no audio, 32.4 MB  
  Source: Eyepetizer (kaiyanapp) <https://www.eyepetizer.net/detail.html?vid=267407>  
  Cut: 51.050–107.950 s of the original, crop [0.000, 0.131, 1.000, 0.869] of the frame  
  sha256 `041eec922ad5c5f26fdd9bd509a57424e77979edb02602cb02443dfd9e1b7eba`; checked by sha256, then content fingerprints  
  Terms: Eyepetizer terms of service; copyright remains with the creator.

### M50_036

- `tasks/M50_036/media/reference.mp4` — video excerpt, 45.9 s, 1280×720, 24 fps, no audio, 23.2 MB  
  Source: Eyepetizer (kaiyanapp) <https://www.eyepetizer.net/detail.html?vid=9616>  
  Cut: 165.050–210.950 s of the original  
  sha256 `de96e6fd8b06739c12f4523e0629d9a89cd995584edfde21fa253638c7291525`; checked by sha256, then content fingerprints  
  Terms: Eyepetizer terms of service; copyright remains with the creator.

### M50_037

- `tasks/M50_037/media/reference.mp4` — video excerpt, 42.9 s, 852×480, 24 fps, no audio, 7.5 MB  
  Source: Sina video "Aladdin parkour (Sina re-upload of a TeamSupertramp film)" <https://edge.ivideo.sina.com.cn/31750298801.mp4?KID=sina,viask&Expires=1792166400&ssig=AR10kckap0&reqid=&r=video.sina.com.cn%2Fview%2F317502988.html> (page: <https://k.sina.cn/article_6137907478_m16dd9091603300owo1.html>)  
  Fallback 1: YouTube "Aladdin Meets Parkour in Real Life (TeamSupertramp)" <https://www.youtube.com/watch?v=JPQNm2d3QIQ>  
  Cut: 145.050–187.950 s of the original  
  sha256 `b93b67a64b10c34a1e252bba1eb3d4b0296688924b37819d54c9307e71151f19`; checked by sha256, then content fingerprints  
  Terms: Sina video terms of service; copyright remains with the rights holder.

### M50_038

- `tasks/M50_038/media/reference.mp4` — video excerpt, 27.9 s, 1280×720, 25 fps, no audio, 11.8 MB  
  Source: YouTube <https://www.youtube.com/watch?v=kXeJAlpbP7U>  
  Cut: 159.050–186.950 s of the original  
  sha256 `5181a72a703d3af95745966e80defd9ff4b9c53d1e8b5cbe3ce36c8fd1b650dd`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### M50_039

- `tasks/M50_039/media/reference.mp4` — video excerpt, 59.9 s, 1920×684, 30 fps, no audio, 29.8 MB  
  Source: Bilibili <https://www.bilibili.com/video/av991820849>  
  Cut: 6.050–65.950 s of the original, crop [0.000, 0.100, 1.000, 0.733] of the frame  
  sha256 `26d4d1d300be1d046d9e5755db7b3be80a70cd9fa8d2a9d30d70b837851ccd82`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### M50_040

- `tasks/M50_040/media/reference.mp4` — video excerpt, 29.4 s, 1920×1080, 24 fps, no audio, 33.1 MB  
  Source: YouTube <https://www.youtube.com/watch?v=Czb_IvmC3ls>  
  Cut: 1200.050–1229.450 s of the original  
  sha256 `91dcec03a82ee10cc982a1382fc7ef434581f913f6edfdeb04bf9f4ab7e33ce9`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### M50_041

- `tasks/M50_041/media/reference.mp4` — video excerpt, 45.9 s, 1280×720, 30 fps, no audio, 29.9 MB  
  Source: YouTube <https://www.youtube.com/watch?v=Yw07ISZbKxQ>  
  Cut: 392.050–437.950 s of the original  
  sha256 `e8905e461344674f1c24a79cae4d6d03f7efb5afc5e950e99407d186ace14b45`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### M50_042

- `tasks/M50_042/media/reference.mp4` — video excerpt, 27.9 s, 1280×720, 59.9 fps, no audio, 20.5 MB  
  Source: YouTube <https://www.youtube.com/watch?v=5Y2LZ15Xb7Y>  
  Cut: 62.050–89.950 s of the original  
  sha256 `0714688f5d4aa45a01a2779e7653b8ea1a3b583594fa0b5057a5ac5a5cfcb085`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### M50_043

- `tasks/M50_043/media/reference.mp4` — video excerpt, 29.9 s, 1920×970, 30 fps, no audio, 43.9 MB  
  Source: Bilibili <https://www.bilibili.com/video/av992653184>  
  Cut: 98.050–127.950 s of the original, crop [0.000, 0.102, 1.000, 1.000] of the frame  
  sha256 `7a5f9e3399b8f9d1fcf14c21596f747c97a1bb3d3e46c7457f27f449dcc5d204`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### M50_044

- `tasks/M50_044/media/reference.mp4` — video excerpt, 52.9 s, 1920×1078, 30 fps, no audio, 41.4 MB  
  Source: Bilibili <https://www.bilibili.com/video/av993676537>  
  Cut: 40.050–92.950 s of the original  
  sha256 `9c66e27e145c7aa106cff2af800ed68fa027fd75e97a1697b2461c12b9807c7d`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### M50_045

- `tasks/M50_045/media/reference.mp4` — video excerpt, 23.4 s, 1280×720, 30 fps, no audio, 11.5 MB  
  Source: YouTube <https://www.youtube.com/watch?v=NOqEfRegnEA>  
  Cut: 907.050–930.450 s of the original  
  sha256 `0759b56695c3c3168d797d14e7e75eda05a8048f75e9e6945ca02c4eace3e5db`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.

### M50_047

- `tasks/M50_047/media/reference.mp4` — video excerpt, 41.9 s, 852×372, 24 fps, no audio, 17.5 MB  
  Source: Bilibili <https://www.bilibili.com/video/av98184777>  
  Cut: 3.050–44.950 s of the original, crop [0.000, 0.113, 1.000, 0.887] of the frame  
  sha256 `5142c31144817bf796401495c875dc964db1023dedd6c74ffceeed418d7a7eee`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Bilibili user agreement; copyright remains with the uploader or original creator.

### M50_049

- `tasks/M50_049/media/reference.mp4` — video excerpt, 44.9 s, 1280×720, 30 fps, no audio, 20.4 MB  
  Source: YouTube <https://www.youtube.com/watch?v=5vUyFekkDTo>  
  Cut: 669.050–713.950 s of the original  
  sha256 `c9e5539b93e887c61085727d2f962d8a15cb6fdae649449f6f67f55a442d12cf`; checked by content fingerprints every 0.5 s (`ok`), byte-identical when the platform serves the same stream  
  Terms: Standard YouTube License; copyright remains with the uploader.
