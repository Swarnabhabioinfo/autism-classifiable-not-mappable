"""
abide2_preprocess.py — preprocess raw ABIDE-II -> Schaefer-100 timeseries.

ABIDE-II is on disk RAW (native-space EPI) in the 96GB One Touch zip. To make it
comparable to the ABIDE-I CPAC data we run a minimal but standard pipeline with
locally-installed FSL: motion correction (mcflirt) -> mean EPI -> BET ->
affine EPI->MNI152-2mm (flirt 12-DOF) -> apply to 4D -> Schaefer-100 parcellation
(nilearn, standardize+detrend). Saves SITE_SUBID_timeseries.npy (ABIDE-I naming).

HONEST CAVEAT: this is lighter than ABIDE-I's full CPAC (no fieldmap/nuisance
regression). It is a REPLICATION cohort; site/pipeline effects are harmonized
downstream with ComBat, and we report the preprocessing difference.

Runs incrementally (skip-if-done), balanced across diagnosis. Background-safe.
"""
import os, re, sys, zipfile, tempfile, subprocess, time
import numpy as np, pandas as pd
from multiprocessing import Pool

ZIP = "/media/swarnabha/One Touch/swarnabha1-20250513_073515.zip"
PHENO = "/DATA/NeuroGen-4D-Starter/Objective 2/ABIDEII_Composite_Phenotypic.csv"
OUT = "/DATA/NeuroGen-4D-Starter/Objective 2/rigorous/abide2_matrices"
MAX = int(os.environ.get("MAX_SUBJ", "300"))     # cap for a timely first cohort
NWORK = int(os.environ.get("NWORK", "6"))
CROP = 200                                        # keep first 200 vols (FC needs no more)
os.makedirs(OUT, exist_ok=True)

os.environ["FSLDIR"] = "/home/swarnabha/fsl"
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
os.environ["PATH"] = "/home/swarnabha/fsl/bin:" + os.environ["PATH"]
MNI = "/home/swarnabha/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz"

_AMAPS = None; _Z = None; _WM = None; _CSF = None
def _init():
    global _AMAPS, _Z, _WM, _CSF
    from nilearn import datasets
    from nilearn.image import resample_to_img
    import nibabel as nib
    a = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
    _AMAPS = a.maps
    _Z = zipfile.ZipFile(ZIP)
    ref = nib.load(MNI)                                   # MNI152_T1_2mm_brain grid
    tp = "/home/swarnabha/fsl/data/standard/tissuepriors"
    wm = resample_to_img(nib.load(f"{tp}/avg152T1_white.img"), ref, interpolation="nearest")
    csf = resample_to_img(nib.load(f"{tp}/avg152T1_csf.img"), ref, interpolation="nearest")
    _WM = np.asarray(wm.dataobj).squeeze() > 0.66         # conservative WM mask (3D)
    _CSF = np.asarray(csf.dataobj).squeeze() > 0.66       # conservative CSF mask (3D)


def sh(cmd):
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process(task):
    sub, site, ent, lab = task
    out_npy = os.path.join(OUT, f"{site}_{sub}_timeseries.npy")
    if os.path.exists(out_npy):
        return (sub, "skip")
    try:
        import nibabel as nib
        with tempfile.TemporaryDirectory() as td:
            raw = os.path.join(td, "rest.nii.gz")
            with _Z.open(ent) as src, open(raw, "wb") as dst:
                dst.write(src.read())
            from nilearn.maskers import NiftiLabelsMasker
            mc = os.path.join(td, "mc"); use = mc
            sh(f'mcflirt -in "{raw}" -out "{mc}" -plots')      # -plots -> mc.par motion
            hdr = nib.load(mc + ".nii.gz"); nv = hdr.shape[3]
            tr = float(hdr.header.get_zooms()[3]); tr = tr if 0.4 < tr < 6 else 2.0
            motion = np.loadtxt(mc + ".par")                    # (nv, 6)
            if nv > CROP:
                use = os.path.join(td, "crop")
                sh(f'fslroi "{mc}" "{use}" 0 {CROP}'); motion = motion[:CROP]
            mean = os.path.join(td, "mean"); brain = os.path.join(td, "brain")
            mat = os.path.join(td, "e2m.mat"); mni = os.path.join(td, "mni")
            sh(f'fslmaths "{use}" -Tmean "{mean}"')
            sh(f'bet "{mean}" "{brain}" -f 0.3')
            sh(f'flirt -in "{brain}" -ref "{MNI}" -omat "{mat}" -dof 12')
            sh(f'flirt -in "{use}" -ref "{MNI}" -applyxfm -init "{mat}" -out "{mni}"')
            # WM/CSF nuisance signals (aCompCor-style mean) from MNI-registered 4D
            d4 = np.asarray(nib.load(mni + ".nii.gz").dataobj)   # (91,109,91,T)
            wm_sig = d4[_WM].mean(0); csf_sig = d4[_CSF].mean(0)  # (T,)
            tissue = np.column_stack([wm_sig, csf_sig])
            # confounds: motion(6)+deriv(6) + WM/CSF(2)+deriv(2); band-pass 0.01-0.1
            conf = np.hstack([motion, np.vstack([np.zeros(6), np.diff(motion, axis=0)]),
                              tissue, np.vstack([np.zeros(2), np.diff(tissue, axis=0)])])
            masker = NiftiLabelsMasker(_AMAPS, standardize=True, detrend=True,
                                       high_pass=0.01, low_pass=0.1, t_r=tr)
            ts = masker.fit_transform(mni + ".nii.gz", confounds=conf)
            np.save(out_npy, ts.astype(np.float32))
        return (sub, f"ok{ts.shape}")
    except Exception as e:
        return (sub, f"FAIL:{str(e)[:60]}")


def build_worklist():
    z = zipfile.ZipFile(ZIP)
    pat = re.compile(r"^([A-Za-z]+)_(\d{5})_(\d)/rest_\d/.*rest\.nii\.gz$")
    entry = {}                                   # subid -> (site, zip_entry)
    for n in z.namelist():
        m = pat.match(n)
        if m:
            site, sub = m.group(1), m.group(2)
            entry.setdefault(sub, (site, n))     # first rest run per subject
    z.close()
    ph = pd.read_csv(PHENO, encoding="latin1")
    ph["SUB_ID"] = ph["SUB_ID"].astype(str).str.replace(".0", "", regex=False)
    dx = dict(zip(ph["SUB_ID"], ph["DX_GROUP"]))
    work = []
    for sub, (site, ent) in entry.items():
        if sub in dx:
            work.append((sub, site, ent, 1 if int(dx[sub]) == 1 else 0))
    asd = [w for w in work if w[3] == 1]; ctrl = [w for w in work if w[3] == 0]
    inter = []
    for a, c in zip(asd, ctrl):
        inter += [a, c]
    inter += asd[len(ctrl):] + ctrl[len(asd):]
    return inter


def main():
    work = build_worklist()[:MAX]
    print(f"[abide2] worklist {len(work)} subjects | {NWORK} workers | crop {CROP} vols", flush=True)
    t0 = time.time(); ok = fail = skip = 0
    with Pool(NWORK, initializer=_init) as pool:
        for i, (sub, status) in enumerate(pool.imap_unordered(process, work), 1):
            if status.startswith("ok"): ok += 1
            elif status == "skip": skip += 1
            else: fail += 1; print(f"  {sub} {status}", flush=True)
            if i % 10 == 0:
                el = time.time() - t0
                print(f"  [{i}/{len(work)}] ok={ok} skip={skip} fail={fail} | "
                      f"{el/60:.1f}min ({el/max(i-skip,1):.0f}s/subj eff)", flush=True)
    n_done = len([f for f in os.listdir(OUT) if f.endswith('.npy')])
    print(f"[abide2] complete: ok={ok} skip={skip} fail={fail} | total .npy={n_done} | "
          f"{(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
