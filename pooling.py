"""Pooling independent within-event scatter: precision versus class mismatch."""
import numpy as np
import clock_model as c

CLASSES=np.arange(len(c.SYNC))%2


def estimate(repeats, labels=None):
    """Center EACH event separately; never pool different timestamp means.

    Pool residual scatter with its degrees of freedom. Equal repeat budgets
    make this the mean of event sample covariances, not covariance of raw data.
    """
    z=np.asarray(repeats,float)
    if z.ndim!=3 or z.shape[2]!=2 or z.shape[1]<3 or not np.all(np.isfinite(z)):
        raise ValueError('Require finite (events,m>=3,2) repeats')
    r=z-z.mean(axis=1,keepdims=True)
    s=np.einsum('nki,nkj->nij',r,r)/(z.shape[1]-1)
    if labels is not None:
        labels=np.asarray(labels)
        if labels.shape!=(len(z),) or labels.dtype.kind not in 'iu':
            raise ValueError('Require one integer quality class per event')
        s=np.stack([s[labels==g].mean(axis=0) for g in labels])
    if np.any(np.linalg.eigvalsh(s)[:,0]<=0):
        raise ValueError('Covariance is singular')
    return s


def covariances(mismatch):
    """Within each known quality class, half have q times larger noise SD."""
    if not np.isfinite(mismatch) or mismatch<1:
        raise ValueError('Require finite mismatch >=1')
    scale=np.where((np.arange(len(CLASSES))//2)%2,mismatch,1.)
    return c.covariances(8)*scale[:,None,None]**2


def summary(e):
    e=np.asarray(e)
    if not len(e):return dict(successful_trials=0)
    return dict(rmse_ms=float(np.sqrt(np.mean(e**2))),bias_ms=float(e.mean()),
                sd_ms=float(e.std(ddof=1)),mean_error_mcse_ms=float(e.std(ddof=1)/np.sqrt(len(e))),successful_trials=len(e))
