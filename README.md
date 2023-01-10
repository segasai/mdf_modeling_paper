This repo has the code used in the paper by Deason, Koposov et al(2023).

The module mdf_decompose allows you to take the list of metallicities and model that as a combination of galaxy luminosities.

The main method is mdf_decompose.doit() (see detailed documentation there)


The shortest example of use:

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
fehs = rstate.normal(size=N) * fehsig + fehmean

ret = M.doit(fehs,
                  curmv,
                  mass_met_info=MMI)
# print the grid 
print (ret['logl_grid'], ret['n50'])