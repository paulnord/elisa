"""Validate complete native spectral fits and collect matched CPU and cuBLAS timings."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.study_native_mesh_refinement import read, sha256, store
import numpy as np


def flux_error(a, b):
    return float(np.abs(a - b).max(initial=0))


def compare_traces(reference, candidate):
    with np.load(reference) as a, np.load(candidate) as b:
        rows = {tuple(x):i for i,x in enumerate(a["positions"])}
        shared = [(rows[tuple(x)], j) for j,x in enumerate(b["positions"]) if tuple(x) in rows]
        ia, ib = np.asarray(shared, dtype=int).reshape(-1,2).T
        checks = {k:bool(np.array_equal(a[k], b[k])) for k in ("positions","chain","accepted")}
        errors = {}
        for k in ("flux", "posterior", "chain_logp"):
            x,y = (a[k],b[k]) if k=="chain_logp" else (a[k][ia],b[k][ib])
            finite = np.isfinite(x)
            valid = finite & np.isfinite(y)
            same = np.array_equal(finite, np.isfinite(y))
            checks[k] = bool(same and np.allclose(x[valid], y[valid],
                              atol=2e-12 if k=="flux" else 1e-6, rtol=3e-13 if k=="flux" else 0))
            errors[k] = flux_error(x[valid],y[valid])
        return dict(all_passed=all(checks.values()), checks=checks, matching_proposal_count=len(ia),
                    max_abs=errors, accepted_per_walker=b["accepted"].tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("native_mesh_results",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    folder=args.folder
    earlier=read(args.native_mesh_results)
    coords=earlier["sampled_points"]
    idx={v["label"]:i for i,v in enumerate(coords)}
    ref=earlier["native_predictions"]["1.5"]
    choice=read(folder/"choice.json") if (folder/"choice.json").exists() else None
    records={}
    for src in sorted(folder.glob("*-case.json")):
        item=read(src)
        if item.get("complete"):
            records[src.name.removesuffix("-case.json")]=item
    # Direct reproductions can use the same result stems without the Ada scheduler.
    for src in sorted(folder.glob("*-a*.json")):
        stem,sep,last=src.stem.rpartition("-a")
        if not sep or not last.isdecimal() or stem in records: continue
        item=read(src)
        if "blackbody_calls_by_process" in item:
            records[stem]=dict(name=stem,output=src.name,valid=True,process_wall_s=None)
    native_name=next(k for k in ("chain-reference-native-8","chain-reference-native-16") if records.get(k,{}).get("valid"))
    def trace(out):
        return out.with_name(out.stem+"-chain")/"trace.npz"
    baseline=trace(folder/records[native_name]["output"])
    with np.load(baseline) as a:
        mapped={tuple(v):i for i,v in enumerate(a["positions"])}
        common=[mapped[tuple(p["position"])] for p in coords[1:]]
        np.testing.assert_allclose(a["flux"][common],ref["flux"][1:],atol=2e-12,rtol=3e-13)
        np.testing.assert_allclose(a["posterior"][common],ref["log_posterior"][1:],atol=1e-6,rtol=0)
        full_reference={k:np.where(np.isfinite(a[k]),a[k],None).tolist() for k in a.files}
    curves,chains,faces,skipped={},{},{},{}
    for name,item in sorted(records.items()):
        if not item.get("valid"):
            skipped[name]={k:item.get(k) for k in ("workers","reason","interruption","available_gib")}
            continue
        path=folder/item["output"];run=read(path)
        assert run["fixture_sha256"]==earlier["fixture_sha256"] and run["primary_degrees"]==1.5
        row={k:run[k] for k in ("action","backend","primary_degrees","complete_wall_s","blackbody_calls_by_process",
                                "cpu_affinity","regular_cpu_threads","gpu")}
        row.update({k:item.get(k) for k in ("process_wall_s","window","checkpoint_unchanged")})
        if run["action"]=="curves":
            errors=[]
            for v in run["rows"]:
                i=idx[v["label"]]
                np.testing.assert_array_equal(v["position"],coords[i]["position"])
                fa,fb=np.asarray(v["flux"]),np.asarray(ref["flux"][i])
                lp=abs(v["posterior"]-ref["log_posterior"][i])
                errors.append(dict(label=v["label"],max_abs_flux=flux_error(fa,fb),abs_logp=lp,
                                   pass_both=bool(np.allclose(fa,fb,atol=2e-12,rtol=3e-13) and lp<=1e-6)))
            row.update(checks=errors,curve_seconds=[v["seconds"] for v in run["rows"]])
            if run["face_check"]:
                row["face_check"]=run["face_check"];faces[name]=row
            else:curves[name]=row
        else:
            c=run["chain"]
            row.update(sampler={k:c[k] for k in ("workers","posterior_calls","physical_likelihoods","nonfinite_posteriors",
                "seconds","sampling_seconds","accepted","proposed_moves","chain_sha256","evaluated_coordinates_sha256",
                "autocorrelation_steps","observed_worker_settings","os_threads_per_walker_process","emcee_proposal_seed",
                "worker_posterior_calls","thread_environment","initial_preflight")})
            row.update(parity=compare_traces(baseline,trace(path)),likelihoods_per_s=c["physical_likelihoods"]/run["complete_wall_s"])
            chains[name]=row
    modes=("blocked","cpu-dgemm","cuda-dgemm")
    def combine_curves(prefix):
        groups={}
        for mode in modes:
            samples=[curves[f"{prefix}-{i}-{mode}"] for i in range(3)]
            checks=[x for s in samples for x in s["checks"]]
            assert len({r["label"] for r in checks})==12
            forward=sum(s["complete_wall_s"] for s in samples)
            groups[mode]=dict(complete_wall_s=forward,likelihoods_per_s=12/forward,
                              max_flux=max(x["max_abs_flux"] for x in checks),max_log_posterior=max(x["abs_logp"] for x in checks),
                              all_passed=all(x["pass_both"] for x in checks) and faces["face-"+mode]["face_check"]["passed"])
        return groups
    groups=combine_curves("curves")
    warm=combine_curves("warm") if any(k.startswith("warm-") for k in curves) else None
    selected=min((m for m in modes[:2] if groups[m]["all_passed"]),key=lambda m:groups[m]["complete_wall_s"],default="native")
    if choice:assert selected==choice["selected_cpu"]
    results=dict(fixture_sha256=earlier["fixture_sha256"],native_mesh_results_sha256=sha256(args.native_mesh_results),
        primary_degrees=1.5,mesh_sigma_limit=0.1,mesh_2_to_1p5=earlier["comparisons"]["2/1.5"],
        limits=dict(flux_atol=2e-12,flux_rtol=3e-13,log_posterior_atol=1e-6,face_atol=1e-10,face_rtol=3e-13),
        frozen_native_reference_checked_at=48,reference_name=native_name,reference_trace=full_reference,
        curve_groups=groups,warm_curve_groups=warm,curve_runs=curves,face_checks=faces,selected_cpu=selected,chains=chains,skipped_cases=skipped)
    store(args.output,results)


if __name__=="__main__":
    main()
