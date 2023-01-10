import numpy as np
import dynesty.utils as dyutil
import multiprocessing as mp
import scipy.special
import dynesty, dynesty.pool
import contextlib
import math

mv_sun = 4.81


class KirbyMassMetInfo:
    """
    Class representing the mass metallicity relation
    """

    def mass_met(log10l):
        """ avg feh at a given log10 (lum)"""
        return -1.68 + 0.29 * (log10l - 6)

    def log10_sig(log10l):
        """ log10 of MDF width at a given log10(lum) """
        return np.log10(0.45 - 0.06 * (log10l - 5))  # np.minimum(logl, -0.2)

    def mass_met_spread():
        """ Spread of the mass metallicity relation"""
        return 0.15


def get_weights(n_inbin, nmax_distinct):
    """
    Populate the 2d occupancy matrix, to pick up random fe/h in each
    logl bin according to the mass metallicity spread
    """
    nbins = len(n_inbin)
    weights = np.zeros((nbins, nmax_distinct), dtype=int)

    xind = n_inbin <= nmax_distinct
    weights[xind, :] = np.arange(nmax_distinct)[None, :] < n_inbin[xind][:,
                                                                         None]
    xind1 = ~xind

    weights[xind1, :] = n_inbin[xind1][:, None] // nmax_distinct + (
        ((n_inbin[xind1] % nmax_distinct)[:, None]) >
        np.arange(nmax_distinct)[None, :]).astype(int)
    assert (weights.sum(axis=1) == n_inbin).all()
    return weights


def evaler(xvals,
           logl_bins,
           n_inbin,
           rng=None,
           feh_means=None,
           nmax_distinct=200,
           mass_met_info=None):
    nbins = len(logl_bins)
    if rng is not None and feh_means is None:
        eps = rng.normal(size=(nbins, nmax_distinct))
        feh_means0 = mass_met_info.mass_met(logl_bins)
        feh_means = feh_means0[:, None] + eps * mass_met_info.mass_met_spread()
    elif feh_means is not None and rng is None:
        nmax_distinct = 1
        pass
    else:
        raise RuntimeError('feh_means and rng are both set or unset')
    feh_widths = 10**(mass_met_info.log10_sig(logl_bins)[:, None])
    nstars = 10**logl_bins
    weights = get_weights(n_inbin, nmax_distinct)
    mult = (weights) * nstars[:, None] / np.sqrt(2 * np.pi) / feh_widths
    # these are multipliers in front of exp(-((feh-mu)/sig)^2)
    # this assumes that feh_width is the same for every galaxy

    # may need to renormalize if considering restricted range of fehs
    # may need to sum to one

    # making all gaussian param 1ds
    mult = mult.reshape(-1)
    feh_means = feh_means.reshape(-1)
    feh_widths = (feh_widths[:, None] + np.zeros(
        (1, nmax_distinct))).reshape(-1)
    xind = mult > 0
    if not xind.any():
        return xvals * 0 - 1e100, -1e100
    logp = (np.log(mult[None, xind]) - 0.5 * ((
        (xvals[:, None] - feh_means[None, xind]) / feh_widths[None, xind])**2))
    ret = scipy.special.logsumexp(logp, axis=1)
    return ret, 0


class CubeTransf:
    """
    Class that is transform the parameter space into unit cube
    needed for nested sampling
    """

    def __init__(
            self,
            maxlogn=3,
            nseed=1000,
            minlogl=1.9,
            maxlogl=8  # logl range for the most massive system)
    ):
        self.minlogl = minlogl
        self.maxlogl = maxlogl
        self.nseed = nseed
        self.maxlogn = maxlogn

    def __call__(self, x):
        """
        Apply the actual cube transform
        """
        # first two dimensions are feh, logl then are the occupation numbers
        # followed by the seed
        nbins = len(x) - 3
        minfeh = -4
        maxfeh = 0.5
        feh = minfeh + (maxfeh - minfeh) * x[0]
        logl = self.minlogl + (self.maxlogl - self.minlogl) * x[1]
        ns = 10**(-1 + (self.maxlogn + 1) * x[2:nbins + 2])
        # seed = x[nbins]
        seed = (x[nbins + 2] * self.nseed)
        return np.concatenate(([feh, logl], ns, [seed]))


def logp(p, data, getModel=False):
    """
    Evaluate the logprobability of data given parameter vector
    """
    (fehs, logl_bins, logl0, loglerr, nmax_distinct, mass_met_info, prior_only,
     minfeh, maxfeh) = data
    nbins = len(logl_bins)
    sections = [2, nbins, 1]
    (feh_ref,
     logl_ref), xparam, seed0 = np.array_split(p,
                                               np.cumsum(sections)[:-1])
    nfehbins = 100
    xvals = np.linspace(minfeh, maxfeh, nfehbins)
    # interpolation grid of metallicities
    n_inbin = xparam.astype(int)
    if seed0 < 0:
        return -1e100
    seed0 = seed0[0]
    seed = seed0.astype(int)
    # seed = [int(_) for _ in (seed0.tobytes())]
    totlum = np.log10(np.sum(n_inbin * 10**logl_bins) + 10**logl_ref)

    # avoid plateau in logl
    penalty = (n_inbin - xparam).sum() * 0.001
    penalty += (seed - seed0) * .001

    penalty += -0.5 * ((totlum - logl0) / loglerr)**2
    penalty += -0.5 * ((feh_ref - mass_met_info.mass_met(logl_ref)) /
                       mass_met_info.mass_met_spread())**2
    if prior_only:
        return penalty
    rng = np.random.default_rng(seed)
    logp0, penalty1 = evaler(xvals,
                             logl_bins,
                             n_inbin,
                             rng=rng,
                             nmax_distinct=nmax_distinct,
                             mass_met_info=mass_met_info)
    penalty += penalty1
    logp_ref, _ = evaler(xvals,
                         np.r_[logl_ref],
                         np.r_[1],
                         feh_means=np.r_[feh_ref],
                         mass_met_info=mass_met_info)
    II = scipy.interpolate.InterpolatedUnivariateSpline(xvals, logp0, k=1)
    II_ref = scipy.interpolate.InterpolatedUnivariateSpline(xvals,
                                                            logp_ref,
                                                            k=1)
    norm = (np.exp(II(xvals)).sum() +
            np.exp(II_ref(xvals)).sum()) * (xvals[1] - xvals[0])
    logp1 = np.logaddexp(II(fehs), II_ref(fehs)) - np.log(norm)
    if getModel:
        return logp1
    return np.sum(logp1) + penalty


def doit(
    fehs,
    curmv,
    npar=None,
    nthreads=36,
    nlive=10000,
    neff=None,
    steps=100,
    rstate=1,
    maxlogn=3,  # max(log(occupation number in each logl bin))
    magpad=2.5,  # how far away to go in luminosity above
    mv_err=0.1,  # luminosity uncertainty
    minlogl=None,  # minimum log(L) in the grid
    maxlogl=None,  # maximum log(L) in the grid
    mass_met_info=None,
    # a class with 3 functions mass_met(log10l), mass_met_spread()
    # and log10_sig(log10l)
    # that return mean feh at log10l, spread in mass met relation and
    # log10 width of the mdf
    prior_only=False,
    minfeh=-4,  # minimum feh in the data/truncation
    maxfeh=0.5  # maximum feh in the data/truncation
):
    """
    Model  the MDF as combination of MDF of lower luminosity
    objects
    
    Parameters
    ----
    fehs: array
        Array of [Fe/H] of modelled stars
    curmv: absolute luminosity of the system
    npar: int or None
        The number of luminosity bins. If none, the number of bins 
        is decided automatically
    nthreads: int
        The number of threads needed for sampling
    nlive: int
        The number of live-points for nested sampling 
    neff: int 
        The desired number of effective samples in the posterior
    steps: int
        The number of MCMC walk steps used at each iteration of 
        nested sampling 
    rstate: int or RandomGenerator
        The initialization of the random state generator
    maxlogn: int
        The maximum value of the log of the occupation number in each
        luminosity bin
    magpad: float
        How much brighter than the luminisity of the system the bins should go
    mv_err: float
        The uncertainty in the MV of the system
    minlogl: float or None
        minimum log(L) in the grid (if None, it's chosen automatically)
    maxlogl: float or None,  
        maximum log(L) in the grid
    mass_met_info: MassMetInfo class or None
        a class instance that should have 
        3 methods mass_met(log10l), mass_met_spread(), and log10_sig(log10l)
        that return mean feh at log10l, spread in mass met relation and
        log10 width of the mdf
    prior_only: boolean
        if True on the the prior is sampled, the data is ignored
    minfeh: float
        The minimum [Fe/H] in the data/truncation
    maxfeh: float
        The maximum [Fe/H] in the data/truncation

    Returns
    ------
    ret: dict 
         Dictionary logl grid and posterior samples
         Keys are 
         logl_grid: the logl grid
         samp: The raw posterior samples
         samp_n: The posterior samples for just the occupation numbers 
               in logl bins
         n_16, cumul_n16, n84...: the percentiles of the occupation 
               numbers  in bins (or for cumulativae number of dwarfs equal
               or brighter than a luminosity of the bin)
    """
    curlogl = -((curmv - mv_sun) / 2.5)
    loglerr = mv_err / 2.5

    if maxlogl is None:
        maxlogl = curlogl + magpad / 2.5
    if minlogl is None:
        minlogl = 1.9
    default_step_mag = 1
    if npar is None:
        npar = math.ceil((maxlogl - minlogl) / (default_step_mag / 2.5))
    # this is to ensure we have a point on curlogl
    #     step = (curlogl - minlogl) / int(npar * (curlogl - minlogl) /
    #                                     (maxlogl - minlogl))
    step = (maxlogl - minlogl) / (npar - 1)
    logl_grid = np.arange(npar) * step + minlogl
    nmax_distinct = 200
    logl_args = (fehs, logl_grid, curlogl, loglerr, nmax_distinct,
                 mass_met_info, prior_only, minfeh, maxfeh)
    # nseed = nlive
    nseed = 2000
    cube_transf = CubeTransf(nseed=nseed,
                             maxlogn=maxlogn,
                             minlogl=minlogl,
                             maxlogl=maxlogl)
    if isinstance(rstate, int):
        rstate = np.random.default_rng(rstate)
    with (dynesty.pool.Pool(
            nthreads, logp, cube_transf, logl_args=(logl_args, ))
          if nthreads > 1 else contextlib.nullcontext()) as pool:
        if nthreads > 1:
            curl, curp, curargs = pool.loglike, pool.prior_transform, None
        else:
            curl, curp, curargs = logp, cube_transf, (logl_args, )
        dns = dynesty.DynamicNestedSampler(
            curl,
            curp,
            npar + 3,
            nlive=nlive,
            rstate=rstate,
            logl_args=curargs,
            pool=pool,
            # sample='unif',
            # sample='rslice',
            walks=steps,
            slices=steps,
            sample='rslice',
        )
        dns.run_nested(n_effective=neff)
    # dlogz_init=0.5, maxbatch=0)
    samp = dyutil.resample_equal(
        dns.results['samples'],
        np.exp(dns.results['logwt'] - dns.results['logz'][-1]))
    xlogl = samp[:, 1] * 1
    xsamp = samp[:, 2:npar + 2].astype(int) * 1
    xind = np.argmin(np.abs(logl_grid[None, :] - xlogl[:, None]), axis=1)
    xsamp[np.arange(len(samp)), xind] += 1
    res = dict(logl_grid=logl_grid, samp=samp, samp_n=xsamp)
    for pp in [1, 5, 16, 50, 84, 95, 99]:
        res['n%d' % pp] = scipy.stats.scoreatpercentile(xsamp, pp, axis=0)
        res['cumul_n%d' % pp] = scipy.stats.scoreatpercentile(np.cumsum(
            xsamp[:, ::-1], axis=1)[:, ::-1],
                                                              pp,
                                                              axis=0)
    return res
