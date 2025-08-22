#!/usr/bin/env python3
"""
search_and_align_single.py

Run MaSIF-search in two stages for a SINGLE-CHAIN query protein against an arbitrary binder database:
  Stage 1: Ultrafast retrieval via nearest-neighbor on MaSIF descriptors
  Stage 2: Alignment + NN reranking (no ground-truth; ranks by ScoreNN)

USAGE
-----
python search_and_align_single.py <data_dir> <query_id> <db_list.txt> <K> <ransac_iter> <top_align>

Args
  data_dir     : Path to the MaSIF data root (e.g., ../../../data/masif_ppi_search)
  query_id     : Query ID as used by MaSIF preprocessing, e.g. PROA_A_  (note trailing underscore)
  db_list.txt  : File with one binder ID per line, each like BIND1_B_ (trailing underscore)
  K            : Number of top descriptor matches (decoys) to keep globally
  ransac_iter  : Number of RANSAC iterations for alignment
  top_align    : Align/rerank this many best decoys (caps the Stage 2 workload)

Expectations
  - Descriptors exist at:
       <data_dir>/<masif_opts["ppi_search"]["desc_dir"]>/<ID>/p1_desc_flipped.npy   (for query)
       <data_dir>/<masif_opts["ppi_search"]["desc_dir"]>/<ID>/p2_desc_straight.npy  (for binders)
  - Surfaces (.ply) exist at:
       <data_dir>/<masif_opts["ply_chain_dir"]>/<CHAIN>.ply   where CHAIN is PDBID_CHAIN
  - Precomputations exist under:
       <data_dir>/<masif_opts["ppi_search"]["masif_precomputation_dir"]>/<ID>/
       <data_dir>/<masif_opts["site"]["masif_precomputation_dir"]>/

Output
  - Prints a ranked table of candidate binders by ScoreNN (higher is better)
  - Writes a lightweight TSV summary to: search_single_results.tsv

Notes
  - This script does not compute RMSD or require co-crystals.
  - If any file is missing for a binder, that binder is skipped with a warning.
"""

import os
import sys
import numpy as np
from collections import defaultdict

# MaSIF imports — must be runnable from repo root or ensure PYTHONPATH points to source/
try:
    from default_config.masif_opts import masif_opts
    from alignment_utils_masif_search import (
        get_center_and_random_rotate,
        get_patch_geo,
        multidock,
        subsample_patch_coords,
    )
    from transformation_training_data.score_nn import ScoreNN
    from geometry.open3d_import import *  # provides read_point_cloud, PointCloud, etc.
    from scipy.spatial import cKDTree
except Exception as e:
    sys.stderr.write("[ERROR] Could not import MaSIF modules. Run from repo root or set PYTHONPATH to source/.\n")
    sys.stderr.write(f"Exception: {e}\n")
    sys.exit(1)


def id_to_chain_name(masif_id: str) -> str:
    """Convert PROA_A_ -> PROA_A (chain basename used for .ply/.pdb filenames)."""
    if masif_id.endswith("_"):
        masif_id = masif_id[:-1]
    return masif_id  # already PDBID_CHAIN


def load_query(data_dir: str, query_id: str):
    desc_dir = os.path.join(data_dir, masif_opts["ppi_search"]["desc_dir"])  
    precomp_dir = os.path.join(data_dir, masif_opts["ppi_search"]["masif_precomputation_dir"])  
    precomp_dir_9A = os.path.join(data_dir, masif_opts["site"]["masif_precomputation_dir"])  
    surf_dir = os.path.join(data_dir, masif_opts["ply_chain_dir"])  

    q_desc_path = os.path.join(desc_dir, query_id, "p1_desc_flipped.npy")
    sc_labels_path = os.path.join(precomp_dir, query_id, "p1_sc_labels.npy")

    if not os.path.isfile(q_desc_path):
        raise FileNotFoundError(f"Missing query descriptors: {q_desc_path}")
    if not os.path.isfile(sc_labels_path):
        raise FileNotFoundError(f"Missing query p1_sc_labels.npy: {sc_labels_path}")

    q_desc = np.load(q_desc_path)
    sc_labels = np.load(sc_labels_path)  # shape ~ (1, Npoints, ?)
    sc_labels = np.nan_to_num(sc_labels[0])
    # center patch = highest median shape complementarity
    center_point = int(np.argmax(np.median(sc_labels, axis=1)))

    # Load target surface point cloud
    chain_name = id_to_chain_name(query_id)
    target_ply = os.path.join(surf_dir, f"{chain_name}.ply")
    if not os.path.isfile(target_ply):
        raise FileNotFoundError(f"Missing target surface: {target_ply}")
    target_pcd = read_point_cloud(target_ply)

    # One-point coord for the target center (geodesic patch will be computed around this point)
    target_coord = subsample_patch_coords(query_id, "p1", precomp_dir_9A, [center_point])

    # Extract geodesic patch and descriptors around the center
    target_patch, target_patch_descs = get_patch_geo(target_pcd, target_coord, center_point, q_desc, flip=True)

    # KDtree for alignment stage
    target_ckdtree = cKDTree(target_patch.points)

    return {
        "desc_all": q_desc,
        "center_point": center_point,
        "center_desc": q_desc[center_point],
        "patch": target_patch,
        "patch_descs": target_patch_descs,
        "ckdtree": target_ckdtree,
    }


def load_binders(data_dir: str, binder_ids):
    desc_dir = os.path.join(data_dir, masif_opts["ppi_search"]["desc_dir"])  
    precomp_dir_9A = os.path.join(data_dir, masif_opts["site"]["masif_precomputation_dir"])  
    surf_dir = os.path.join(data_dir, masif_opts["ply_chain_dir"])  

    binders = []
    for bid in binder_ids:
        b_desc_path = os.path.join(desc_dir, bid, "p2_desc_straight.npy")
        chain_name = id_to_chain_name(bid.split("_", 1)[0] + "_" + bid.split("_", 1)[1]) if bid.endswith("_") else id_to_chain_name(bid)
        # safer: build chain from ID directly
        chain_name = id_to_chain_name(bid)
        b_ply = os.path.join(surf_dir, f"{chain_name}.ply")
        try:
            if not os.path.isfile(b_desc_path):
                raise FileNotFoundError(f"Missing binder descriptors: {b_desc_path}")
            if not os.path.isfile(b_ply):
                raise FileNotFoundError(f"Missing binder surface: {b_ply}")
            b_desc = np.load(b_desc_path)
            b_pcd = read_point_cloud(b_ply)
            b_coords = subsample_patch_coords(bid, "p2", precomp_dir_9A)
            binders.append({
                "id": bid,
                "desc": b_desc,
                "pcd": b_pcd,
                "coords": b_coords,
            })
        except Exception as e:
            sys.stderr.write(f"[WARN] Skipping binder {bid}: {e}\n")
            continue
    return binders


def stage1_retrieval(query_center_desc, binders, K):
    """Return top-K global decoys across all binders by Euclidean distance to query center patch descriptor.
    Returns list of tuples: (global_rank, binder_ix, binder_patch_index, distance)
    """
    all_dists = []
    all_meta = []  # (binder_ix, vix)
    for bix, b in enumerate(binders):
        d = np.linalg.norm(b["desc"] - query_center_desc, axis=1)  # (Nb_patches,)
        all_dists.append(d)
        all_meta.append(np.stack([np.full(d.shape, bix), np.arange(d.shape[0])], axis=1))
    if not all_dists:
        return []
    dcat = np.concatenate(all_dists, axis=0)
    mcat = np.concatenate(all_meta, axis=0)  # shape (Ntotal, 2)
    order = np.argsort(dcat)
    chosen = order[:K]
    out = []
    for rank, idx in enumerate(chosen):
        bix, vix = int(mcat[idx,0]), int(mcat[idx,1])
        out.append((rank+1, bix, vix, float(dcat[idx])))
    return out


def stage2_align_and_score(target, binders, decoys, ransac_iter, max_to_align):
    """Run alignment on top decoys and score with ScoreNN.
    Returns list of dicts with keys: binder_id, score, distance, binder_patch, global_rank.
    """
    nn_model = ScoreNN()
    results = []
    # Optionally cap the number of decoys to align for speed
    decoys_to_run = decoys[:max_to_align]
    for (global_rank, bix, vix, dist) in decoys_to_run:
        b = binders[bix]
        # Prepare a random-centered copy of binder PCD for robustness
        b_pcd = copy.deepcopy(b["pcd"]) if "copy" in globals() else PointCloud(b["pcd"])  # fallback
        try:
            random_T = get_center_and_random_rotate(b_pcd)
            b_pcd.transform(random_T)
            # Align the single matched binder patch index (vix) to the target patch
            all_results, all_source_patch, all_scores = multidock(
                b_pcd,
                b["coords"],
                b["desc"],
                np.array([vix]),
                target["patch"],
                target["patch_descs"],
                target["ckdtree"],
                nn_model,
                ransac_iter=ransac_iter,
            )
            # Keep the best score among hypotheses for this decoy
            if len(all_scores) > 0:
                best_ix = int(np.argmax(all_scores))
                best_score = float(all_scores[best_ix])
                results.append({
                    "binder_id": b["id"],
                    "score": best_score,
                    "distance": dist,
                    "binder_patch": int(vix),
                    "global_rank": int(global_rank),
                })
        except Exception as e:
            sys.stderr.write(f"[WARN] Alignment failed for {b['id']} patch {vix}: {e}\n")
            continue
    # Sort by ScoreNN (desc)
    results.sort(key=lambda x: (-x["score"], x["distance"]))
    return results


def main():
    if len(sys.argv) != 7:
        print(__doc__)
        sys.exit(1)
    data_dir, query_id, db_list_file, K, ransac_iter, top_align = sys.argv[1:]
    K = int(K); ransac_iter = int(ransac_iter); top_align = int(top_align)

    # Load IDs
    if not os.path.isfile(db_list_file):
        sys.stderr.write(f"[ERROR] db_list.txt not found: {db_list_file}\n")
        sys.exit(1)
    binder_ids = [ln.strip() for ln in open(db_list_file) if ln.strip()]

    # Load query
    print(f"[INFO] Loading query: {query_id}")
    target = load_query(data_dir, query_id)
    print(f"  center_point={target['center_point']}")

    # Load binders
    print(f"[INFO] Loading {len(binder_ids)} binders...")
    binders = load_binders(data_dir, binder_ids)
    print(f"  usable binders: {len(binders)}")
    if len(binders) == 0:
        sys.stderr.write("[ERROR] No usable binders.\n")
        sys.exit(1)

    # Stage 1: retrieval
    print(f"[INFO] Stage 1: ranking top {K} decoys by descriptor distance...")
    decoys = stage1_retrieval(target["center_desc"], binders, K)
    if len(decoys) == 0:
        sys.stderr.write("[ERROR] No decoys found.\n")
        sys.exit(1)
    print("  Top-10 decoys (global_rank, binder_id, binder_patch, distance):")
    for r, bix, vix, d in decoys[:10]:
        print(f"    {r:3d}  {binders[bix]['id']:>12s}  {vix:6d}  {d:8.3f}")

    # Stage 2: alignment + ScoreNN reranking
    print(f"[INFO] Stage 2: aligning top {min(top_align, len(decoys))} decoys (RANSAC iters={ransac_iter})...")
    results = stage2_align_and_score(target, binders, decoys, ransac_iter, top_align)

    # Report
    print("\n=== FINAL RANKING (by ScoreNN) ===")
    for i, r in enumerate(results[:20], 1):
        print(f"{i:3d}. {r['binder_id']:>12s}  score={r['score']:.4f}  d(desc)={r['distance']:.3f}  patch={r['binder_patch']}  (from decoy rank {r['global_rank']})")

    # Save a TSV summary
    out_path = "search_single_results.tsv"
    with open(out_path, "w") as f:
        f.write("rank\tbinder_id\tscore\tdesc_distance\tbinder_patch\tdecoy_rank\n")
        for i, r in enumerate(results, 1):
            f.write(f"{i}\t{r['binder_id']}\t{r['score']:.6f}\t{r['distance']:.6f}\t{r['binder_patch']}\t{r['global_rank']}\n")
    print(f"\n[INFO] Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
