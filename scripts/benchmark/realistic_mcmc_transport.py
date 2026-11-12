"""Complete physical ELISA fits: bound-method tasks vs initialized workers.

Benchmark only. Production source files are never edited. The prototype changes
only the pool initializer and the likelihood callback in MCMCFit._fit at runtime.
Synthetic observations are generated with ELISA's real physical LC/RV engines.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import multiprocessing as mp
import os
import pickle
import platform
import statistics
import subprocess
import sys
import textwrap
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from elisa import LCData, RVData, Observer, BinarySystem, settings, units as u
from elisa.analytics.binary_fit import mcmc, mcmc_diagnostics, mixins
from elisa.analytics.models import lc, rv
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.observer import mp_manager

BASELINE = '6dc055ba989ecb558c5df0530bee00a57a155549'
BANDS = ('Generic.Bessell.V', 'Generic.Bessell.B')
N_WALKERS = 32
_CONTEXT = None
_WORKER_FIT = None
_ACTIVE = None
_ORIGINAL_FIT = mcmc.MCMCFit._fit
_ORIGINAL_POOL = mcmc._mcmc_pool
_ORIGINAL_INITIALIZER = mcmc._initialize_mcmc_worker
_ORIGINAL_WORKER = mixins.MCMCMixin.worker


def initialize_original(values, diagnostics):
    _ORIGINAL_INITIALIZER(values, diagnostics)
    logging.disable(logging.CRITICAL)


def initialize_evaluator(values, diagnostics, fitter):
    global _WORKER_FIT
    initialize_original(values, diagnostics)
    _WORKER_FIT = fitter


def evaluate_worker(vector):
    return _WORKER_FIT.ln_probability(vector)


class RecordingPool:
    def __init__(self, pool):
        self.pool = pool
        self.maps = []

    def map(self, function, vectors):
        started = time.perf_counter()
        result = self.pool.map(function, vectors)
        self.maps.append({'points': len(vectors), 'seconds': time.perf_counter() - started})
        return result


@contextmanager
def baseline_pool(diagnostics):
    with _ORIGINAL_POOL(diagnostics) as pool:
        yield RecordingPool(pool) if pool is not None else None


@contextmanager
def initialized_pool(diagnostics, fitter):
    if settings.NUMBER_OF_MCMC_PROCESSES <= 1:
        yield None
        return
    snapshot = copy.copy(fitter)
    snapshot.last_sampler = None  # results belong to the parent, not likelihood evaluation
    with _CONTEXT.Pool(
        processes=settings.NUMBER_OF_MCMC_PROCESSES,
        initializer=initialize_evaluator,
        initargs=(settings.settings_serializer(), diagnostics, snapshot),
    ) as pool:
        yield RecordingPool(pool)


def timed_worker(*args, **kwargs):
    started = time.perf_counter()
    result = _ORIGINAL_WORKER(*args, **kwargs)
    if _ACTIVE is not None:
        _ACTIVE['sampling_seconds'] = time.perf_counter() - started
        pool = args[0].pool
        _ACTIVE['maps'] = [] if pool is None else pool.maps
    return result


def make_prototype():
    source = textwrap.dedent(inspect.getsource(_ORIGINAL_FIT))
    before = '"log_prob_fn": self.ln_probability,'
    after = '"log_prob_fn": evaluate_worker if settings.NUMBER_OF_MCMC_PROCESSES > 1 else self.ln_probability,'
    assert source.count(before) == 1
    assert source.count('_mcmc_pool(diagnostics) as pool:') == 1
    source = source.replace(before, after).replace(
        '_mcmc_pool(diagnostics) as pool:', 'initialized_pool(diagnostics, self) as pool:')
    mcmc.__dict__['evaluate_worker'] = evaluate_worker
    mcmc.__dict__['initialized_pool'] = initialized_pool
    namespace = {}
    exec(compile('from __future__ import annotations\n' + source, '<benchmark-only prototype>', 'exec'), mcmc.__dict__, namespace)
    return namespace['_fit'], source


def configure(context):
    global _CONTEXT
    _CONTEXT = mp.get_context(context)
    # All diagnostics and pools must create their synchronized state in one context.
    mcmc.get_mp_context = lambda: _CONTEXT
    mcmc_diagnostics.get_mp_context = lambda: _CONTEXT
    mp_manager.get_mp_context = lambda: _CONTEXT
    mcmc._initialize_mcmc_worker = initialize_original
    mcmc._mcmc_pool = baseline_pool
    mixins.MCMCMixin.worker = staticmethod(timed_worker)
    settings.configure(
        NUMBER_OF_PROCESSES=1, NUMBER_OF_MCMC_PROCESSES=2,
        NUMBER_OF_THREADS=1, MESH_GENERATOR='auto', LIMB_DARKENING_LAW='linear',
        REFLECTION_EFFECT=True, REFLECTION_EFFECT_ITERATIONS=2,
        USE_SINGLE_LD_COEFFICIENTS=False, MCMC_DIAGNOSTICS_INTERVAL_STEPS=100,
        PENALIZE_ALL_EXCEPTIONS=False, SUPPRESS_WARNINGS=True,
    )
    logging.disable(logging.CRITICAL)


def parameters_from_truth(truth):
    return {group: {name: {'value': value, 'fixed': True} for name, value in values.items()}
            for group, values in truth.items()}


def make_lc_problem():
    truth = {
        'system': {'inclination':85.0, 'eccentricity':0.0, 'argument_of_periastron':0.0,
                   'period':3.0, 'mass_ratio':0.7, 'semi_major_axis':10.0},
        'primary': {'t_eff':6500.0, 'surface_potential':6.0, 'gravity_darkening':0.32,
                    'albedo':0.5, 'metallicity':0.0},
        'secondary': {'t_eff':5000.0, 'surface_potential':6.0, 'gravity_darkening':0.32,
                      'albedo':0.5, 'metallicity':0.0},
    }
    cfg = parameters_from_truth(truth)
    for group, name, lower, upper in (
        ('system','inclination',82.,88.), ('primary','t_eff',6200.,6800.),
        ('secondary','t_eff',4700.,5300.), ('primary','surface_potential',5.7,6.3),
    ):
        cfg[group][name] = {'value':truth[group][name], 'fixed':False, 'min':lower, 'max':upper}
    cfg['nuisance'] = {'ln_f':{'value':-7., 'fixed':True}}
    flat = {f'{group}@{name}':value for group, values in truth.items() for name,value in values.items()}
    flat.update({f'{component}@atmosphere':'ck04' for component in ('primary','secondary')})
    rng = np.random.default_rng(71633)
    data = {}
    for band, count in zip(BANDS, (600, 500), strict=True):
        phases = np.sort(rng.uniform(0.,1.,count))
        observer = Observer(passband=[band], system=None)
        observer.system_cls = BinarySystem
        curve = lc.synthetic_binary(phases, 5.0, observer, **flat)[band]
        values = curve * (1. + rng.normal(0., .002, count))
        data[band] = LCData(x_data=phases, y_data=values, y_err=.002*curve,
                            x_unit=u.dimensionless_unscaled, y_unit=u.dimensionless_unscaled)
    fitter = mcmc.DetachedLightCurveFit()
    fitter.atmosphere_model = {'primary':'ck04','secondary':'ck04'}
    fitter.limb_darkening_coefficients = None
    options = {'discretization':5.0, 'interp_treshold':120, 'samples':'uniform'}
    metadata = {'model':'detached circular eclipsing binary', 'truth':truth,
                'bands':list(BANDS), 'observations':{b:len(d.x_data) for b,d in data.items()},
                'atmosphere':'CK04', 'limb_darkening':'Van Hamme, linear, per-face',
                'reflection_iterations':2, 'surface_discretization_degrees':5.0,
                'uniform_interpolation_threshold':120, 'free_parameters':4,
                'data':'deterministic physical synthetic observations with 0.2% Gaussian noise'}
    return fitter,cfg,data,options,metadata


def make_rv_problem():
    truth = {'eccentricity':.15, 'argument_of_periastron':25., 'period':3.,
             'mass_ratio':.7, 'asini':10., 'gamma':12000.}
    cfg = parameters_from_truth({'system':truth})
    for name,lower,upper in [('gamma',8000.,16000.),('asini',9.,11.),('mass_ratio',.6,.8),('eccentricity',.1,.2)]:
        cfg['system'][name] = {'value':truth[name], 'fixed':False, 'min':lower, 'max':upper}
    cfg['nuisance'] = {'ln_f':{'value':-7., 'fixed':True}}
    rng = np.random.default_rng(4181)
    data = {}
    for component,count in [('primary',300),('secondary',250)]:
        xs = np.sort(rng.uniform(0.,1.,count))
        obs = Observer(passband='bolometric',system=None)
        obs.system_cls = RadialVelocitySystem
        values = rv.central_rv_synthetic(xs,obs,**{f'system@{k}':v for k,v in truth.items()})[component]
        data[component] = RVData(x_data=xs,y_data=values+rng.normal(0.,200.,count),
            y_err=np.full(count,200.),x_unit=u.dimensionless_unscaled,y_unit=u.m/u.s)
    return mcmc.CentralRadialVelocity(),cfg,data,{}, {'model':'double-lined eccentric kinematic RV',
        'truth':truth,'observations':{c:len(d.x_data) for c,d in data.items()},'free_parameters':4,
        'data':'deterministic physical synthetic observations with 200 m/s Gaussian noise'}


def initial_state(fitter,cfg,data,options):
    # This order is defined by ELISA itself, not by guessed parameter ordering.
    if isinstance(fitter,mcmc.LightCurveFit):
        fitter.set_up(BinaryInitialParameters(**copy.deepcopy(cfg)),data,passband=data.keys(),
            morphology=fitter.MORPHOLOGY,observer_system_cls=BinarySystem,**options)
    else:
        fitter.set_up(BinaryInitialParameters(**copy.deepcopy(cfg)),data,observer_system_cls=RadialVelocitySystem)
    fitter.prepare_priors()
    ndim = len(fitter.initial_vector)
    assert ndim == 4
    p0 = fitter.norm_init_vector + np.random.default_rng(1984).normal(0.,.008,(N_WALKERS,ndim))
    assert np.all((p0>0)&(p0<1))
    return p0


def digest(array):
    a = np.asarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()


def run_fit(fitter,cfg,data,options,p0,steps,burn,variant,prototype,seed):
    global _ACTIVE
    mcmc.MCMCFit._fit = prototype if variant=='initialized' else _ORIGINAL_FIT
    settings.configure(NUMBER_OF_MCMC_PROCESSES=1 if variant=='serial' else 2)
    record = {'variant':variant,'nsteps':steps,'burn_in':burn}
    np.random.seed(seed)
    _ACTIVE = record
    start = time.perf_counter()
    try:
        result = fitter.fit(data=data,x0=BinaryInitialParameters(**copy.deepcopy(cfg)),
            nwalkers=N_WALKERS,nsteps=steps,burn_in=burn,initial_state=p0.copy(),
            progress=False,save=False,**options)
    finally:
        record['wall_seconds'] = time.perf_counter()-start
        _ACTIVE = None
    chain = fitter.last_sampler.get_chain().copy()
    log_probability = fitter.last_sampler.get_log_prob().copy()
    assert np.all(np.isfinite(chain)) and np.all(np.isfinite(log_probability))
    record.update(chain_sha256=digest(chain),log_probability_sha256=digest(log_probability),
                  acceptance_mean=float(np.mean(fitter.last_sampler.acceptance_fraction)),
                  final_result=result)
    if record['maps']:
        record['pool_map_seconds'] = sum(x['seconds'] for x in record['maps'])
        record['steady_half_step_median_seconds'] = statistics.median(x['seconds'] for x in record['maps'][3:])
    return record,chain,log_probability


def grow_retained_chain(fitter,steps):
    """Explicit size stress: tile REAL samples, do not claim a longer sampled run."""
    backend = fitter.last_sampler.backend
    chain,log_prob = backend.get_chain(),backend.get_log_prob()
    nrepeat = math.ceil(steps/len(chain))
    backend.chain = np.tile(chain,(nrepeat,1,1))[:steps].copy()
    backend.log_prob = np.tile(log_prob,(nrepeat,1))[:steps].copy()
    backend.iteration = steps
    assert backend.blobs is None


def pickle_info(fitter):
    # emcee's own pickle implementation drops its terminated pool reference.
    payload = pickle.dumps(fitter.ln_probability,protocol=pickle.HIGHEST_PROTOCOL)
    snapshot = copy.copy(fitter)
    snapshot.last_sampler = None
    return {'baseline_callable_bytes':len(payload),
            'initialized_snapshot_bytes':len(pickle.dumps(snapshot,protocol=pickle.HIGHEST_PROTOCOL)),
            'parameter_vector_bytes':len(pickle.dumps(np.zeros(4),protocol=pickle.HIGHEST_PROTOCOL))}


def environment():
    return {'python':sys.version,'platform':platform.platform(),
        'cpu_count':os.cpu_count(),'affinity':list(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        'cpu_model': next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                           if line.startswith('model name')), platform.processor()) if Path('/proc/cpuinfo').exists() else platform.processor(),
        'packages':{name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','astropy','numba','emcee']},
        'thread_environment':{key:os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']},
        'baseline':BASELINE,'benchmark_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}


def summarize(records):
    summaries=[]
    keys=sorted({(r['case'],r['history']) for r in records if r['variant']!='serial'})
    for case,history in keys:
        rows=[r for r in records if r['case']==case and r['history']==history]
        base=[r for r in rows if r['variant']=='baseline']
        opt=[r for r in rows if r['variant']=='initialized']
        pairs=[(b,next(o for o in opt if o['round']==b['round'])) for b in base]
        summaries.append({'case':case,'history':history,'pairs':len(pairs),
            'baseline_wall_median_s':statistics.median(r['wall_seconds'] for r in base),
            'initialized_wall_median_s':statistics.median(r['wall_seconds'] for r in opt),
            'paired_speedup_median':statistics.median(b['wall_seconds']/o['wall_seconds'] for b,o in pairs),
            'paired_speedup_range':[min(b['wall_seconds']/o['wall_seconds'] for b,o in pairs),max(b['wall_seconds']/o['wall_seconds'] for b,o in pairs)],
            'baseline_sampling_median_s':statistics.median(r['sampling_seconds'] for r in base),
            'initialized_sampling_median_s':statistics.median(r['sampling_seconds'] for r in opt),
            'chain_and_log_probability_exact':True})
    return summaries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--context',choices=['fork','spawn'],required=True)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--lc-steps',type=int,default=12)
    parser.add_argument('--rv-steps',type=int,default=120)
    parser.add_argument('--output',default='benchmark-results')
    args=parser.parse_args()
    assert args.rounds>=2 and args.rounds%2==0
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    # Only benchmark files may differ from the source audited in the conversation.
    assert not subprocess.check_output(['git','diff',BASELINE,'--','src/elisa'],text=True).strip()
    configure(args.context)
    prototype,source=make_prototype()
    (out/'prototype-fit.py.txt').write_text(source)
    result={'completed':False,'context':args.context,'environment':environment(),
            'nwalkers':N_WALKERS,'workers':2,'rounds':args.rounds,'cases':{},'records':[],
            'notes':['full physical synthetic fits, not observational-data science results',
                     'short chains are for timing, not evidence of MCMC convergence',
                     'retained_10000 is a size stress made by tiling actual prior samples',
                     'wall time includes setup, preflight, worker startup, burn-in, sampling, and result summarization',
                     'observation generation, template cloning, and chain persistence are excluded',
                     'save=False; the only code variation is worker state/callback transport']}
    def write(): (out/'results.json').write_text(json.dumps(result,indent=2,default=str))
    try:
        for name,factory,steps,burn in [('lc',make_lc_problem,args.lc_steps,4),('rv',make_rv_problem,args.rv_steps,10)]:
            print('BUILDING_CASE',name,flush=True)
            settings.configure(NUMBER_OF_MCMC_PROCESSES=1)
            fresh,cfg,data,options,metadata=factory()
            p0=initial_state(fresh,cfg,data,options)
            result['cases'][name]=metadata
            # Warm parent numerical paths and compile reusable native disk caches.
            for vector in p0[:2]: assert np.isfinite(fresh.ln_probability(vector))
            # Real prior run; neither timing nor claims pretend it is 10,000 steps.
            prior=copy.deepcopy(fresh)
            prior_record,_,_=run_fit(prior,cfg,data,options,p0,8,burn,'serial',prototype,6026)
            result['cases'][name]['actual_prior_steps']=8
            result['cases'][name]['prior_serial_wall_seconds']=prior_record['wall_seconds']
            enlarged=copy.deepcopy(prior)
            grow_retained_chain(enlarged,10000)
            templates={'fresh':fresh,'reused_actual':prior,'retained_10000':enlarged}
            result['cases'][name]['payloads']={key:pickle_info(value) for key,value in templates.items()}
            # Same seed and starting walkers for every condition, including serial.
            serial,ref_chain,ref_log=run_fit(copy.deepcopy(fresh),cfg,data,options,p0,steps,burn,'serial',prototype,19302)
            serial.update(case=name,history='fresh',round=-1)
            result['records'].append(serial)
            np.savez_compressed(out/f'{name}-serial-reference.npz',chain=ref_chain,log_probability=ref_log)
            print('SERIAL_REFERENCE',name,serial['wall_seconds'],flush=True)
            for round_idx in range(args.rounds):
                histories=list(templates)
                if round_idx%2: histories.reverse()
                for history in histories:
                    variants=['baseline','initialized'] if round_idx%2==0 else ['initialized','baseline']
                    for variant in variants:
                        fitter=copy.deepcopy(templates[history])
                        record,chain,log_prob=run_fit(fitter,cfg,data,options,p0,steps,burn,variant,prototype,19302)
                        np.testing.assert_array_equal(chain,ref_chain)
                        np.testing.assert_array_equal(log_prob,ref_log)
                        assert record['final_result']==serial['final_result']
                        assert not mp.active_children(), 'Worker processes survived a completed fit'
                        record.update(case=name,history=history,round=round_idx)
                        result['records'].append(record)
                        write()
                        print('MEASUREMENT',json.dumps({k:record[k] for k in ['case','history','variant','round','wall_seconds','sampling_seconds']})+' exact=True',flush=True)
            result['summaries']=summarize(result['records']);write()
        result['completed']=True
        result['summaries']=summarize(result['records']);write()
        print('BENCHMARK_SUMMARY_BEGIN',flush=True)
        print(json.dumps({'completed':True,'context':args.context,'environment':result['environment'],
                          'cases':result['cases'],'summaries':result['summaries'],
                          'serial_wall_s':{r['case']:r['wall_seconds'] for r in result['records'] if r['variant']=='serial'}},indent=2),flush=True)
        print('BENCHMARK_SUMMARY_END',flush=True)
    except BaseException:
        result['failure']=traceback.format_exc();write();raise


if __name__=='__main__':main()
