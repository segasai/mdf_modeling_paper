This repo has the code implementing an MDF modeling algorithm presented
in the paper by [Deason, Koposov et al.(2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.6091D/abstract)

The module mdf_decompose allows you to take the list of metallicities and model that as a combination of MDFs of galaxies of different luminosities.

The main method is the mdf_decompose.doit() (see detailed documentation there)

The code depends on numpy, scipy and dynesty modules. Make sure you use the latest version of dynesty

If you have any questions, feel free to send an email to skoposov AT ed DOT ac DOT uk.

The shortest example of use:
```python

import mdf_decompose as M


rstate = np.random.default_rng(1)
N = 100 # 100 stars
MMI = M.KirbyMassMetInfo # Use mass-metallicity from Kirby

# assign a random luminosity to the object
logl = rstate.uniform(3, 6)
curmv = -2.5 * logl + M.mv_sun
# just generate a Gaussian distributed array of metallicities
fehmean = MMI.mass_met(logl) + rstate.normal() * MMI.mass_met_spread()
fehsig = 10**MMI.log10_sig(logl)
fehs = rstate.normal(size=N) * fehsig + fehmean #
# the array of stellar metallicities we are going to model

ret = M.doit(fehs,
                curmv,
                mass_met_info=MMI,
		nlive=100)
# print the grid of log luminosities and
# the 16/50/84th percentiles of number of galaxies contributing to
# each bin
print (ret['logl_grid'], ret['n16'], ret['n50'], ret['n84'])
```
