"""Independent predictive tuning of partial covariance pooling; equal budgets."""
from pathlib import Path
import json
import numpy as np
import optics as o
import clock_model as c
from pooling import CLASSES, covariances, estimate, summary

SEED = 20261009
WEIGHTS = np.linspace(0., 1., 5)  # weight ON the class covariance
LEVELS = (1., 1.5, 2., 4.)
TRAIN, VALIDATE = 6, 6


def mix(event, pooled, weight):
    """Convex combinations preserve positive definiteness, including endpoints."""
    event, pooled = np.asarray(event, float), np.asarray(pooled, float)
    if event.shape != pooled.shape or event.ndim != 3 or event.shape[1:] != (2,2):
        raise ValueError('Require matching (events,2,2) covariance arrays')
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Weight must be in [0,1]')
    for s in (event, pooled):
        if not np.all(np.isfinite(s)) or not np.allclose(s,s.transpose(0,2,1)) or np.any(np.linalg.eigvalsh(s)<=0):
            raise ValueError('Require finite symmetric positive definite covariance')
    return (1-weight)*event+weight*pooled


def differences(repeats):
    """Disjoint pairs remove unknown event means without fitting them.

    d=(w_1-w_0)/sqrt(2) has covariance C for independent equal-noise repeats.
    Do NOT form overlapping differences: that would introduce dependence.
    """
    z=np.asarray(repeats,float)
    if z.ndim != 3 or z.shape[2] != 2 or z.shape[1]<2 or z.shape[1]%2 or not np.all(np.isfinite(z)):
        raise ValueError('Require finite (events,even repeats>=2,2)')
    return (z[:,1::2]-z[:,::2])/np.sqrt(2.)


def score(cov, d):
    """Twice mean Gaussian negative log density, omitting constant 2 log(2 pi)."""
    cov=mix(cov,cov,0.)
    d=np.asarray(d,float)
    if d.ndim!=3 or d.shape[0]!=len(cov) or d.shape[2]!=2 or d.shape[1]<1 or not np.all(np.isfinite(d)):
        raise ValueError('Require finite differences with matching event count')
    # solve avoids explicitly computing matrix inverses.
    solved=np.linalg.solve(cov[:,None],d[...,None])[...,0]
    return float(np.mean(np.linalg.slogdet(cov)[1][:,None]+np.sum(d*solved,axis=-1)))


def tune(training, validation):
    """No truth, final timestamps or optical images enter this selector.

    Exact score ties prefer stronger pooling (WEIGHTS ascending, reverse argmin).
    Training covariance is retained after selection; no post-selection refit.
    """
    event=estimate(training);pooled=estimate(training,CLASSES)
    candidates=[mix(event,pooled,w) for w in WEIGHTS]
    d=differences(validation)
    scores=np.array([score(s,d) for s in candidates])
    index=len(WEIGHTS)-1-int(np.argmin(scores[::-1]))
    return candidates[index],float(WEIGHTS[index]),scores


def experiment(trials=300, seed=SEED):
    if not isinstance(trials,int) or trials<2:raise ValueError('trials must be >=2')
    ir,er,tr,vr=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    truth=np.column_stack((c.SYNC,c.RATE*c.SYNC+c.OFFSET))
    center=o.ORIGIN+o.VELOCITY*(c.RATE*(c.FRAME+c.DELAY)+c.OFFSET)
    centers=np.array([o.fit_center(o.image(center)+ir.normal(0,o.READ_SD,33)) for _ in range(trials)])
    evaluation=er.normal(size=(trials,40,2))
    training=tr.normal(size=(trials,40,TRAIN,2))
    validation=vr.normal(size=(trials,40,VALIDATE,2))
    rows=[]
    for q in LEVELS:
        cov=covariances(q);chol=np.linalg.cholesky(cov)
        errors={k:[] for k in ['known_covariance','pool_12','event_12','pool_6','event_6','tuned_6_plus_6']}
        chosen=[]
        for k,u in enumerate(centers):
            z=truth+np.einsum('nij,nj->ni',chol,evaluation[k])
            train=truth[:,None]+np.einsum('nij,nkj->nki',chol,training[k])
            valid=truth[:,None]+np.einsum('nij,nkj->nki',chol,validation[k])
            selected,w,_=tune(train,valid);chosen.append(w)
            all_repeats=np.concatenate([train,valid],axis=1)
            policies=dict(known_covariance=cov,pool_12=estimate(all_repeats,CLASSES),event_12=estimate(all_repeats),
                pool_6=estimate(train,CLASSES),event_6=estimate(train),tuned_6_plus_6=selected)
            for name,s in policies.items():
                error=1000*(c.delay_from_center(u,c.calibrate(*z.T,s))-c.DELAY)
                errors[name].append(error)
        paired={}
        for name in ('pool_12','event_12'):
            # Paired squared-error differences, not independent-RMSE error bars.
            d=np.square(errors['tuned_6_plus_6'])-np.square(errors[name])
            paired[name]=dict(mse_difference_ms2=float(d.mean()),mcse_ms2=float(d.std(ddof=1)/np.sqrt(trials)))
        rows.append(dict(q=q,selection_counts={str(w):chosen.count(w) for w in WEIGHTS},
                         methods={k:summary(v) for k,v in errors.items()},paired_risk=paired))
    return dict(seed=seed,trials=trials,training_pairs=40*TRAIN,validation_pairs=40*VALIDATE,
                equal_total_pairs=40*(TRAIN+VALIDATE),evaluation_pairs=40,failed_fits=0,
                sigma_px=o.SIGMA,center_sd_px=float(centers.std(ddof=1)),rows=rows)


def demo(out='outputs'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=experiment();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for name,label in [('known_covariance','Known covariance'),('pool_12','Pool: all 12'),('event_12','Per event: all 12'),('tuned_6_plus_6','Tune: 6 + 6')]:
        axes[0].plot(LEVELS,[r['methods'][name]['rmse_ms'] for r in report['rows']],'o-',label=label)
    axes[0].set(title='Equal budget: 480 calibration pairs',ylabel='Camera-delay RMSE (ms)')
    for w in WEIGHTS:
        axes[1].plot(LEVELS,[r['selection_counts'][str(w)]/report['trials'] for r in report['rows']],'o-',label=f'Pool weight {w:g}')
    axes[1].set(title='Independent predictive tuning',ylabel='Selection fraction',ylim=(-.03,1.03))
    for name,label in [('pool_12','Versus pool 12'),('event_12','Versus per event 12')]:
        axes[2].errorbar(LEVELS,[r['paired_risk'][name]['mse_difference_ms2'] for r in report['rows']],
            yerr=[2*r['paired_risk'][name]['mcse_ms2'] for r in report['rows']],fmt='o-',capsize=3,label=label)
    axes[2].axhline(0,color='k',linestyle=':')
    axes[2].set(title='Tuned minus fixed risk; ±2 MCSE',ylabel='Paired MSE difference (ms²)')
    for ax in axes:
        ax.set(xlabel='Hidden within-class SD ratio q');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'experiment.png',dpi=160);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__=='__main__':demo()
