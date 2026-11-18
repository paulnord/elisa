"""Validate the fixed-coordinate native mesh experiment and summarize its numerical evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.study_native_mesh_refinement import first_batch, injection, read, sha256, store, study_positions
import numpy as np


def tag(value):
    return f"{value:g}".replace(".", "p")


def comparison(a, b, sigma, observations):
    diff = np.asarray([r["flux"] for r in a]) - np.asarray([r["flux"] for r in b])
    scaled = diff / sigma
    distances = np.linalg.norm(scaled, axis=1)
    logp = np.abs(np.asarray([r["posterior"] for r in a]) - [r["posterior"] for r in b])
    ss = np.sum(scaled[1:]**2, axis=0)
    order = np.argsort(ss)[::-1][:4]
    return dict(centre_distance=float(distances[0]), proposal_endpoint_count=len(a)-1,
                passing_endpoints=int(np.sum(distances[1:] <= 0.1)), centre_passes=bool(distances[0] <= 0.1),
                endpoint_distance_range=[float(v) for v in (distances[1:].min(), np.median(distances[1:]), distances[1:].max())],
                max_abs_flux=float(np.abs(diff).max()), endpoint_abs_logp_range=[float(logp[1:].min()),float(logp[1:].max())],
                leading_observations=[dict(**observations[i], share_of_squared_change=float(ss[i]/ss.sum())) for i in order])


def summarize_spectra(folder, reference, fixture_hash, degrees):
    """Compare paired fresh processes against each other and the archived native sweep."""
    source_hash = sha256(Path(__file__).with_name("benchmark_native_blackbody_blocks.py"))
    by_name = {row["label"]: row for row in reference}
    def load_run(batch, mode):
        run = read(folder/f"spectrum-{batch}-{mode}.json")
        assert run["driver_sha256"] == source_hash and run["fixture_sha256"] == fixture_hash
        assert run["mode"] == mode and run["primary_degrees"] == degrees
        assert run["blackbody_calls"] > 0
        return run

    pairs = []
    for batch in range(3):
        original, blocked = [load_run(batch, mode) for mode in ("original", "blocked")]
        assert original["blackbody_calls"] == blocked["blackbody_calls"]
        assert original["cpu_affinity"] == blocked["cpu_affinity"]
        errors = []
        for a, b in zip(original["rows"], blocked["rows"], strict=True):
            ref = by_name[a["label"]]
            assert a["label"] == b["label"] and a["position"] == b["position"] == ref["position"]
            np.testing.assert_allclose(a["flux"], ref["flux"], atol=2e-12, rtol=3e-13)
            np.testing.assert_allclose(a["posterior"], ref["posterior"], atol=1e-6, rtol=0)
            err = np.max(np.abs(np.asarray(a["flux"]) - b["flux"]))
            logerr = abs(a["posterior"] - b["posterior"])
            errors.append(dict(label=a["label"], max_abs_flux=float(err), abs_log_posterior=logerr,
                               passed=bool(np.allclose(a["flux"], b["flux"], atol=2e-12, rtol=3e-13) and logerr <= 1e-6)))
        pairs.append(dict(order=["blocked", "original"] if batch == 1 else ["original", "blocked"],
                          original=original, blocked=blocked, numerical_checks=errors,
                          speedup=original["fit_and_likelihoods_wall_s"]/blocked["fit_and_likelihoods_wall_s"]))
    assert len({v["label"] for pair in pairs for v in pair["numerical_checks"]}) == 12
    face_check = load_run(3, "radiance-check")["radiance_check"]
    assert face_check["values"] > 0
    times = {mode:sum(pair[mode]["fit_and_likelihoods_wall_s"] for pair in pairs) for mode in ("original", "blocked")}
    return dict(driver_sha256=source_hash, primary_degrees=degrees, pairs=pairs, radiance_check=face_check,
                limits=dict(flux_atol=2e-12,flux_rtol=3e-13,log_posterior_atol=1e-6,radiance_atol=1e-10,radiance_rtol=3e-13),
                fit_and_likelihoods_wall_s=times, likelihoods_per_s={k:12/v for k,v in times.items()},
                overall_speedup=times["original"]/times["blocked"],
                all_checks_passed=bool(face_check["passed"] and all(r["passed"] for p in pairs for r in p["numerical_checks"])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fixture", type=Path)
    p.add_argument("evidence", type=Path)
    p.add_argument("cpu_reference", type=Path, help="original completed m032 CPU fit and its trace.npz")
    p.add_argument("output", type=Path)
    args = p.parse_args()
    folder = args.evidence
    # Also accept standalone reproductions made with the shell commands in the report.
    if (folder/"selection.json").is_file():
        selection = read(folder/"selection.json")
    else:
        levels = sorted({read(path)["primary_degrees"] for path in folder.glob("points-*.json")},reverse=True)
        pilots = sorted({read(path)["primary_degrees"] for path in folder.glob("centre-*.json")},reverse=True)
        selection = dict(sweep_levels=levels, all_levels=pilots, sweep_finest=min(levels), injection_degrees=min(pilots))
    levels = selection["sweep_levels"]
    finest = min(levels)
    injected = selection["injection_degrees"]
    fixture = read(args.fixture)
    driver_hash = sha256(Path(__file__).with_name("study_native_mesh_refinement.py"))
    positions, walkers, factors, threshold = study_positions(fixture)
    nominal = None
    measured, decisions, scheduling = {}, {}, {}
    for degrees in levels:
        centre = read(folder/f"centre-{tag(degrees)}.json")
        scheduling[degrees] = dict(pilot_s=centre["rows"][0]["seconds"], pilot_rss_mib=centre["linux_peak_rss_mib"])
        assert centre["fixture_sha256"] == sha256(args.fixture)
        sections = sorted(folder.glob(f"points-{tag(degrees)}-*.json"))
        inputs = [centre, *(read(path) for path in sections)]
        assert all(d["primary_degrees"] == degrees and d["fixture_sha256"] == centre["fixture_sha256"]
                   and d["driver_sha256"] == driver_hash for d in inputs)
        labels = {r["label"]: r for d in inputs for r in d["rows"]}
        assert len(labels) == sum(len(d["rows"]) for d in inputs) == len(positions)
        rows = [labels[name] for name, _ in positions]
        for item, (_, coord) in zip(rows, positions, strict=True):
            np.testing.assert_array_equal(item["position"], coord)
        assert np.isfinite([r["flux"] for r in rows]).all() and np.isfinite([r["posterior"] for r in rows]).all()
        if nominal is None:
            nominal = np.asarray(centre["nominal_sigma"])
        for d in inputs:
            np.testing.assert_array_equal(d["nominal_sigma"], nominal)
        posterior = np.asarray([r["posterior"] for r in rows])
        margin = factors + posterior[33:] - posterior[1+walkers] - threshold
        decisions[degrees] = dict(accepted=(margin>0).tolist(), margin=margin.tolist())
        measured[degrees] = rows
    assert len(walkers)==16 and nominal.size==28
    coarse_inj = injection(fixture,read(folder/"centre-12.json"))
    for band in fixture["data"]:
        np.testing.assert_allclose(coarse_inj["data"][band]["y"],fixture["data"][band]["y"],atol=2e-12,rtol=3e-13)

    old = read(args.fixture.with_name("mcmc-scaling-mesh.json"))
    assert old["trace_sha256"] == sha256(args.cpu_reference/"trace.npz")
    assert old["fixture_sha256"] == sha256(args.fixture)
    assert [r["index"] for r in old["rows"]] == list(range(old["evaluated"]))
    with np.load(args.cpu_reference/"trace.npz") as trace:
        mapping = {tuple(pos):i for i,pos in enumerate(trace["positions"])}
        ii = [mapping[tuple(pos)] for _,pos in positions[1:]]
        for i, idx in enumerate(ii, 1):
            r = measured[12][i]
            np.testing.assert_allclose(r["flux"], trace["flux"][idx],atol=2e-12,rtol=3e-13)
            np.testing.assert_allclose(r["posterior"], trace["posterior"][idx],atol=1e-6,rtol=0)
            for j, d in enumerate((12,8)):
                delta=measured[d][i]["log_likelihood"]-measured[6][i]["log_likelihood"]
                np.testing.assert_allclose(delta,old["rows"][idx]["delta_log_likelihood_to_finest"][j],atol=1e-6,rtol=0)
        initial=np.asarray(fixture["initial"])
        w,q,_,_=first_batch(fixture)
        masks=np.array(decisions[12]["accepted"])
        np.testing.assert_array_equal(trace["chain"][0,w], np.where(masks[:,None],q,initial[w]))

    obs=[dict(band=k,phase=float(v)) for k, data in fixture["data"].items() for v in data["x"]]
    pairs={}
    for i,a in enumerate(levels):
        for b in levels[i+1:]:
            result=comparison(measured[a],measured[b],nominal,obs)
            ea,eb=[np.asarray(decisions[d]["accepted"]) for d in (a,b)]
            result.update(flipped_walkers=walkers[ea!=eb].tolist(),flips=int((ea!=eb).sum()))
            pairs[f"{a:g}/{b:g}"]=result
    targets=np.array([r["flux"] for r in measured[finest]])
    both=np.ones(len(targets),dtype=bool)
    for deg in levels[-3:-1]:
        alt=np.array([r["flux"] for r in measured[deg]])
        both &= np.linalg.norm((alt-targets)/nominal,axis=1)<=0.1
    centre_reference=read(folder/f"centre-{tag(injected)}.json")["rows"][0]
    pilot_distances={}
    for path in folder.glob("centre-*.json"):
        centre=read(path);row=centre["rows"][0]
        pilot_distances[f"{centre['primary_degrees']:g}"]=dict(primary_secondary_degrees=row["primary_secondary_degrees"],
            joint_sigma_to_finest_centre=float(np.linalg.norm((np.array(row["flux"])-centre_reference["flux"])/nominal)),
            delta_log_posterior_to_finest_centre=row["posterior"]-centre_reference["posterior"],
            rss_mib=centre["linux_peak_rss_mib"],cold_forward_seconds=row["seconds"])
    external=None
    if (folder/"post-holdout-0p9.json").is_file():
        post=read(folder/"post-holdout-0p9.json");row=post["rows"][0]
        assert row["position"] == fixture["centre"] and post["driver_sha256"] == driver_hash
        external=dict(primary_secondary_deg=row["primary_secondary_degrees"],rss_mib=post["linux_peak_rss_mib"],
                      cold_s=row["seconds"],centre_joint_sigma={})
        for path in folder.glob("centre-*.json"):
            a=read(path)
            external["centre_joint_sigma"][f"{a['primary_degrees']:g}"]=float(
                np.linalg.norm((np.array(a["rows"][0]["flux"])-row["flux"])/nominal))
    fits,steps={},{}
    for path in sorted(folder.glob("recover-*.json")):
        out=read(path)
        assert out["driver_sha256"] == driver_hash and out["fixture_sha256"] == sha256(args.fixture)
        fits[f"{out['primary_degrees']:g}/{out['injection_degrees']:g}"]=out
    for path in sorted(folder.glob("step-check-*.json")):
        out=read(path);d=out["primary_degrees"];base=fits[f"{d:g}/{injected:g}"]
        shift=(np.asarray(out["fit"])-base["fit"])/base["conditional_sigma"]
        steps[tag(d)]=dict(fit=out["fit"],solver_status=out["solver_status"],active_bounds=out["active_bounds"],
                           chi2=out["chi2_fit"],signed_shift_over_formal_sigma=shift.tolist())
    profiles={d:read(folder/f"profile-{tag(d)}.json") for d in dict.fromkeys((6,3,finest))}
    for d in profiles:
        profile=profiles[d]
        assert profile["profiled_flux_equal"] and profile["stages"]["compute_surface_coverage"]>0
        assert profile["stages"]["apply_reflection_effect"]>0
    save=dict(model_commit="a9f0a9205674b61df0f1d18a645d9df9294417f7",fixture_sha256=sha256(args.fixture),
              driver_sha256=driver_hash,
              reference_trace_sha256=old["trace_sha256"],old_flux_posterior_and_first_decisions_verified=True,
              old_12_8_6_density_differences_verified=True,coarse_injection_reconstructed=True,
              limits=dict(nominal_joint_sigma=0.1,flux_parity_atol=2e-12,flux_parity_rtol=3e-13,posterior_parity_atol=1e-6),
              hardware=dict(host="Ada",cpu="AMD Ryzen 9 5950X, 16 cores and 32 logical threads",gpu="NVIDIA GeForce RTX 3080 Ti, 12 GiB, hidden for CPU study",
                            numeric_threads=1,curve_processes=1,mcmc_processes=1,process_affinity="one CPU per child"),
              selection=selection,actual_angles={d:rows[0]["primary_secondary_degrees"] for d,rows in measured.items()},
              finest_triple=dict(primary_degrees=levels[-3:],passing_sampled_endpoints=int(both[1:].sum()),
                                 total_sampled_endpoints=48,centre_passed=bool(both[0])),
              centre_holdout=pilot_distances,independent_post_holdout=external,
              observations=obs,conditional_parameters=["inclination","primary surface potential","secondary surface potential"],
              sampled_points=[dict(label=k,position=v) for k,v in positions],nominal_sigma=nominal.tolist(),
              native_predictions={d:dict(flux=[r["flux"] for r in rows],
                                         log_likelihood=[r["log_likelihood"] for r in rows],
                                         log_posterior=[r["posterior"] for r in rows]) for d,rows in measured.items()},
              samplers=dict(seed=fixture["move_seed"],walkers=walkers.tolist(),stretch_log_factors=factors.tolist(),
                            log_uniforms=threshold.tolist(),decisions=decisions),
              comparisons=pairs,conditional_fits=fits,half_step_controls=steps,profile=profiles,
              crossed_stars=read(folder/"crossed.json") if (folder/"crossed.json").exists() else None,
              supplementary_crossed=[read(p) for p in sorted(folder.glob("crossed-extra-*.json"))],
              blackbody_batching=(summarize_spectra(folder,measured[finest],sha256(args.fixture),finest)
                                  if (folder/"spectrum-0-original.json").exists() else None),
              resource_pilots=scheduling)
    store(args.output,save)


if __name__=="__main__":
    main()
