"""Download author Zhuang archives, join real CCF coordinates, and export offline 3D HTML.

Example:
    python scripts/zhuang_ccf_3d.py --download --dataset 2

Only AnnData observation metadata is read; expression matrices stay on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import zipfile
import zlib

DATASETS = {
    2: {"size": 376293233, "sha256": "d6ac8a4607dc31fd1423fe95adead7639da6b5bcccc68aeda738d3d839d47c32"},
    3: {"size": 542553210, "sha256": "f19d281077659c22e36e61c007e556bcf5f74e7d61df8575424daee8b392a1ef"},
}
PROJECT = Path(__file__).resolve().parents[1]
SOURCE = "https://huggingface.co/datasets/gkrieg13/mimyr-data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(data_dir: Path, dataset: int = 2) -> Path:
    """Resume a task-owned .part file; publish ZIP only after size/SHA-256 verification."""
    import requests
    from tqdm.auto import tqdm

    expected = DATASETS[dataset]
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / f"Zhuang-ABCA-{dataset}.zip"
    if target.exists():
        if target.stat().st_size != expected["size"] or sha256(target) != expected["sha256"]:
            raise ValueError(f"Existing ZIP failed verification; preserving it: {target}")
        print(f"Verified existing archive: {target}")
        return target
    partial = target.with_suffix(".zip.part")
    start = partial.stat().st_size if partial.exists() else 0
    if start > expected["size"]:
        raise ValueError(f"Partial download is larger than the expected archive: {partial}")
    if shutil.disk_usage(data_dir).free < expected["size"] - start + 64 * 1024**2:
        raise OSError("Not enough disk space for the ZIP download.")
    if start < expected["size"]:
        url = f"{SOURCE}/resolve/main/{target.name}?download=true"
        headers = {"Range": f"bytes={start}-"} if start else {}
        with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            if "text/html" in response.headers.get("Content-Type", ""):
                raise ValueError("Received HTML instead of the ZIP archive.")
            if response.status_code == 206:
                if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise ValueError("Server returned a different byte range; partial file was preserved.")
            elif response.status_code == 200:
                start = 0  # A server that ignores Range requires restarting this partial download.
            else:
                raise ValueError(f"Unexpected download status: {response.status_code}")
            with partial.open("ab" if start else "wb") as stream:
                with tqdm(total=expected["size"], initial=start, unit="B", unit_scale=True,
                          desc=target.name, mininterval=5) as progress:
                    for chunk in response.iter_content(2 * 1024**2):
                        if chunk:
                            stream.write(chunk)
                            progress.update(len(chunk))
    if partial.stat().st_size != expected["size"] or sha256(partial) != expected["sha256"]:
        raise ValueError(f"Download failed size/SHA-256 verification; partial file retained: {partial}")
    partial.replace(target)
    print("SHA-256 verified:", expected["sha256"])
    return target


def _file_crc32(path: Path) -> int:
    crc = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def extract_archive(archive: Path, data_dir: Path, dataset: int = 2) -> Path:
    """Extract the expected dataset and CSV; preserve matching existing files."""
    from tqdm.auto import tqdm

    archive, data_dir = Path(archive), Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = data_dir / f"Zhuang-ABCA-{dataset}"
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        missing = []
        for member in members:
            target = (data_dir / member.filename).resolve()
            if not target.is_relative_to(data_dir):
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"ZIP contains a symlink: {member.filename}")
            if member.is_dir():
                continue
            if not (target.is_relative_to(dataset_dir)
                    or target == data_dir / f"ccf-ABCA-{dataset}.csv"):
                raise ValueError(f"Unexpected ZIP member: {member.filename}")
            if target.exists():
                if target.stat().st_size != member.file_size or _file_crc32(target) != member.CRC:
                    raise ValueError(f"Existing extracted file differs; preserving it: {target}")
            else:
                missing.append(member)
        required = sum(member.file_size for member in missing)
        if shutil.disk_usage(data_dir).free < required + 64 * 1024**2:
            raise OSError(f"Need {required / 1024**3:.2f} GiB free for extraction.")
        with tqdm(total=required, unit="B", unit_scale=True, desc="Extracting", mininterval=5) as progress:
            for member in missing:
                target = data_dir / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".extracting")
                try:
                    with bundle.open(member) as source, partial.open("wb") as output:
                        for chunk in iter(lambda: source.read(4 * 1024**2), b""):
                            output.write(chunk)
                            progress.update(len(chunk))
                    partial.replace(target)  # ZipExtFile verifies CRC at end of each member.
                finally:
                    if partial.exists():
                        partial.unlink()
    if not dataset_dir.is_dir() or not (data_dir / f"ccf-ABCA-{dataset}.csv").is_file():
        raise ValueError("Archive is missing the section directory or CCF coordinate CSV.")
    return dataset_dir


def join_ccf(obs, ccf, slice_name: str):
    """Join by full string cell IDs, never row order or floating-point conversion."""
    import numpy as np
    import pandas as pd

    if not obs.index.is_unique or not ccf.index.is_unique:
        raise ValueError("Cell IDs must be unique in the section and CCF table.")
    if not {"x", "y", "z"}.issubset(ccf.columns):
        raise ValueError("CCF table must contain x, y, z.")
    observed = obs.copy()
    observed.index = observed.index.astype(str)
    coords = ccf.reindex(observed.index)[["x", "y", "z"]].astype(float)
    valid = np.isfinite(coords.to_numpy()).all(axis=1)
    joined = coords.loc[valid].rename(columns={c: f"{c}_ccf" for c in "xyz"})
    joined.insert(0, "cell_id", joined.index)
    joined["slice"] = slice_name
    # Keep the released atlas class labels and colors; no biological relabeling.
    joined["class"] = (observed.loc[valid, "class"].astype("string").fillna("Unlabeled")
                       if "class" in observed else "Unlabeled")
    joined["class_color"] = (observed.loc[valid, "class_color"].astype("string").fillna("#8c8c8c")
                             if "class_color" in observed else "#8c8c8c")
    for name in ("class", "class_color", "slice"):
        joined[name] = joined[name].astype("category")
    audit = {"slice": slice_name, "n_input": len(observed), "n_ccf": len(joined),
             "n_without_ccf": len(observed) - len(joined),
             "ccf_fraction": len(joined) / len(observed) if len(observed) else 0.0}
    return joined.reset_index(drop=True), audit


def load_ccf_cells(data_dir: Path, dataset: int = 2, slices: list[int] | None = None):
    """Return all matched cells plus per-section audit, without reading expression X."""
    import anndata as ad
    import pandas as pd
    from tqdm.auto import tqdm

    data_dir = Path(data_dir)
    folder = data_dir / f"Zhuang-ABCA-{dataset}"
    ccf_file = data_dir / f"ccf-ABCA-{dataset}.csv"
    if not ccf_file.is_file():
        raise FileNotFoundError(f"Download/extract the author archive first: {ccf_file}")
    ccf = pd.read_csv(ccf_file, dtype={"cell_label": str},
                      usecols=["cell_label", "x", "y", "z"]).set_index("cell_label")
    if not ccf.index.is_unique:
        raise ValueError("The CCF CSV contains duplicate cell_label values.")
    paths = sorted(folder.glob("sec_*.h5ad")) if slices is None else [folder / f"sec_{i:03}.h5ad" for i in slices]
    if not paths or any(not p.is_file() for p in paths):
        raise FileNotFoundError("Section files are missing; check --data-dir and --slices.")
    tables, audits = [], []
    for path in tqdm(paths, desc="Joining real CCF coordinates", mininterval=5):
        section = ad.read_h5ad(path, backed="r")
        try:
            columns = [name for name in ["class", "class_color"] if name in section.obs]
            table, audit = join_ccf(section.obs[columns], ccf, path.stem)
        finally:
            section.file.close()
        tables.append(table)
        audits.append(audit)
    cells = pd.concat(tables, ignore_index=True)
    if cells.empty:
        raise ValueError("No selected cells have real CCF coordinates; choose registered interior sections.")
    if cells["cell_id"].duplicated().any():
        raise ValueError("Duplicate cell IDs across selected sections.")
    for name in ("class", "class_color", "slice"):
        cells[name] = cells[name].astype("category")
    return cells, pd.DataFrame(audits)


def visualization_sample(cells, max_points: int = 120000, seed: int = 42):
    """Uniform reproducible cell sample; plotting alone is subsampled."""
    if max_points < 1:
        raise ValueError("max_points must be positive.")
    if len(cells) <= max_points:
        return cells.copy().reset_index(drop=True)
    return cells.sample(n=max_points, random_state=seed).sort_index().reset_index(drop=True)


INTERACTIVE_JS = r"""
(function () {
  const d = __PAYLOAD__;
  const plot = document.getElementById('{plot_id}');
  const panel = document.createElement('section');
  panel.style.cssText = 'padding:16px 24px;background:#f5f7fa;font:14px Arial,sans-serif;display:flex;gap:18px;align-items:center;flex-wrap:wrap';
  plot.parentNode.insertBefore(panel, plot);
  function selector(label, values) {
    const wrapper = document.createElement('label');
    wrapper.textContent = label + ' ';
    const select = document.createElement('select');
    select.style.cssText = 'font:14px Arial;padding:5px;max-width:320px';
    for (const [value, text] of values) {
      const option = document.createElement('option');
      option.value = value; option.textContent = text; select.appendChild(option);
    }
    wrapper.appendChild(select); panel.appendChild(wrapper); return select;
  }
  const slice = selector('切片 / Section', [['all','全部'], ...d.slices.map((x,i)=>[String(i),x])]);
  const cellClass = selector('细胞类型 / Class', [['all','全部'], ...d.classes.map((x,i)=>[String(i),x])]);
  const color = selector('着色 / Color', [['class','细胞类型'],['slice','切片']]);
  const caption = document.createElement('span'); panel.appendChild(caption);
  const reset = document.createElement('button'); reset.textContent='重置视角'; panel.appendChild(reset);
  const palette = ['#4477aa','#ee6677','#228833','#ccbb44','#66ccee','#aa3377','#bbbbbb','#332288','#88ccee','#117733','#ddcc77','#cc6677'];
  const layout = plot.layout;
  layout.uirevision = 'keep-camera';
  async function render() {
    const names = color.value === 'class' ? d.classes : d.slices;
    const codes = color.value === 'class' ? d.class_codes : d.slice_codes;
    const groups = new Map();
    for (let i=0; i<d.x.length; i++) {
      if (slice.value !== 'all' && d.slice_codes[i] !== Number(slice.value)) continue;
      if (cellClass.value !== 'all' && d.class_codes[i] !== Number(cellClass.value)) continue;
      const k=codes[i];
      if (!groups.has(k)) groups.set(k, {type:'scatter3d',mode:'markers',name:names[k],x:[],y:[],z:[],customdata:[],
        marker:{size:1.5,opacity:0.65,color:color.value==='class'?d.class_colors[k]:palette[k%palette.length]},
        hovertemplate:'%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]}<br>CCF x=%{x:.3f} mm<br>CCF y=%{y:.3f} mm<br>CCF z=%{z:.3f} mm<extra></extra>'});
      const trace=groups.get(k);
      trace.x.push(d.x[i]); trace.y.push(d.y[i]); trace.z.push(d.z[i]);
      trace.customdata.push([d.ids[i],d.slices[d.slice_codes[i]],d.classes[d.class_codes[i]]]);
    }
    const traces = [...groups.entries()].sort((a,b)=>a[0]-b[0]).map(x=>x[1]);
    const visible = traces.reduce((total,t)=>total+t.x.length,0);
    caption.textContent = '当前显示 '+visible.toLocaleString()+' / 抽样 '+d.x.length.toLocaleString()+' / 完整 CCF '+d.total.toLocaleString()+' 个细胞';
    await Plotly.react(plot, traces, layout, {responsive:true,displaylogo:false,scrollZoom:true});
    plot.dataset.ccfReady = 'true';
    plot.dataset.visiblePoints = String(visible);
  }
  slice.addEventListener('change',render);
  cellClass.addEventListener('change',render);
  color.addEventListener('change',render);
  reset.addEventListener('click',()=>Plotly.relayout(plot,{'scene.camera':d.camera}));
  render().catch(error=>{caption.textContent='绘图失败：'+error.message;console.error(error);});
})();
"""


def write_3d_html(sample, output: Path, dataset: int = 2, total_cells: int | None = None) -> Path:
    """Offline Plotly figure with section/class filters; CCF axes remain in mm."""
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    if sample.empty:
        raise ValueError("Cannot visualize an empty sample.")
    classes = sorted(sample["class"].astype(str).unique())
    slices = sorted(sample["slice"].astype(str).unique())
    class_codes = pd.Categorical(sample["class"].astype(str), categories=classes).codes
    slice_codes = pd.Categorical(sample["slice"].astype(str), categories=slices).codes
    colors = sample.groupby("class", observed=True)["class_color"].first().astype(str).to_dict()
    # Validate label colors rather than interpolating untrusted CSS values.
    import re
    palette = [colors.get(name, "#8c8c8c") for name in classes]
    palette = [color if re.fullmatch(r"#[0-9a-fA-F]{6}", color) else "#8c8c8c" for color in palette]
    camera = {"eye": {"x": 1.55, "y": 1.25, "z": 1.1}, "up": {"x": 0, "y": -1, "z": 0}}
    payload = {"x": sample["x_ccf"].tolist(), "y": sample["y_ccf"].tolist(),
               "z": sample["z_ccf"].tolist(), "ids": sample["cell_id"].astype(str).tolist(),
               "class_codes": class_codes.tolist(), "slice_codes": slice_codes.tolist(),
               "classes": classes, "slices": slices, "class_colors": palette,
               "total": len(sample) if total_cells is None else int(total_cells), "camera": camera}
    xyz = sample[["x_ccf", "y_ccf", "z_ccf"]].to_numpy(dtype=float)
    if not np.isfinite(xyz).all():
        raise ValueError("Visualization requires finite CCF coordinates.")
    axis = lambda name, j: {"title": f"CCF {name} (mm)",
                             "range": [float(xyz[:, j].min()), float(xyz[:, j].max())],
                             "backgroundcolor": "#fafbfc", "gridcolor": "#dde3ea"}
    fig = go.Figure(layout={
        "title": {"text": f"Zhuang-ABCA-{dataset} · Registered cells in Allen CCF", "x": 0.03},
        "height": 780, "margin": {"l": 10, "r": 10, "t": 65, "b": 15},
        "font": {"family": "Times New Roman, Arial, sans-serif", "size": 14},
        "paper_bgcolor": "white", "legend": {"itemsizing": "constant"},
        "scene": {"xaxis": axis("x", 0), "yaxis": axis("y", 1), "zaxis": axis("z", 2),
                  "aspectmode": "data", "camera": camera},
    })
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    script = INTERACTIVE_JS.replace("__PAYLOAD__", encoded)
    html = fig.to_html(full_html=True, include_plotlyjs=True, post_script=script,
                       config={"responsive": True, "displaylogo": False, "scrollZoom": True},
                       div_id="zhuang-ccf-3d")
    html = html.replace('<html>', '<html lang="zh-CN">', 1).replace('<head>', '<head><meta charset="utf-8">', 1)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=int, choices=sorted(DATASETS), default=2)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    parser.add_argument("--download", action="store_true", help="Download/verify/extract before plotting.")
    parser.add_argument("--slices", type=int, nargs="+", help="Section numbers, e.g. 31 32 33 34 35.")
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_points < 1:
        parser.error("--max-points must be positive")
    if args.download:
        archive = download_archive(args.data_dir, args.dataset)
        extract_archive(archive, args.data_dir, args.dataset)
    cells, audit = load_ccf_cells(args.data_dir, args.dataset, args.slices)
    sample = visualization_sample(cells, args.max_points, args.seed)
    output = args.output or PROJECT / f"results/zhuang_ccf_3d/Zhuang-ABCA-{args.dataset}_ccf3d.html"
    write_3d_html(sample, output, args.dataset, len(cells))
    audit.to_csv(output.with_suffix(".audit.csv"), index=False)
    summary = {"dataset": args.dataset, "source": SOURCE, "n_registered_cells": len(cells),
               "n_visualized_cells": len(sample), "visualization_seed": args.seed,
               "coordinate_units": "mm", "slices": args.slices,
               "archive_sha256": DATASETS[args.dataset]["sha256"]}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(audit.to_string(index=False))
    print(f"Registered cells: {len(cells):,}; displayed: {len(sample):,}; HTML: {output.resolve()}")


if __name__ == "__main__":
    main()
