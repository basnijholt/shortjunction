"""Functions library used to calculate phase diagrams for the "Robustness of
Majorana bound states in the short junction limit" paper by 
Doru Sticlet, Bas Nijholt, and Anton Akhmerov

arXiv:1609.00637, to be published in PRB."""


# 1. Standard library imports
from itertools import product
import subprocess
from types import SimpleNamespace

# 2. External package imports
from discretizer import Discretizer, momentum_operators
import holoviews as hv
import ipyparallel
import kwant
import numpy as np
import scipy.sparse.linalg as sla
from scipy.constants import hbar, m_e, eV, physical_constants
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from sympy.physics.quantum import TensorProduct as kr
import sympy

# 3. Internal imports
from wraparound import wraparound


sx, sy, sz = [sympy.physics.matrices.msigma(i) for i in range(1, 4)]
s0 = sympy.eye(2)


class SimpleNamespace(SimpleNamespace):
    """Updates types.SimpleNamespace to have a .update() method.
    Useful for parallel calculation."""
    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        return self


def get_git_revision_hash():
    """Get the git hash to save with data to ensure reproducibility."""
    git_output = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    return git_output.decode("utf-8").replace('\n', '')


# Parameters taken from arXiv:1204.2792
# All constant parameters, mostly fundamental constants, in a SimpleNamespace.
constants = SimpleNamespace(
    m=0.015 * m_e, # effective mass in kg
    g=50, # Lande factor
    hbar=hbar,
    m_e=m_e,
    eV=eV,
    meV=eV * 1e-3)

constants.t = (constants.hbar ** 2 / (2 * constants.m)) * (1e18 / constants.meV)  # meV * nm^2
constants.mu_B = physical_constants['Bohr magneton'][0] / constants.meV
constants.delta_2d = constants.hbar**2 * np.pi**2 / (8 * (100e-9)**2 * constants.m) / constants.meV
constants.unit_B = 2 * constants.delta_2d / (constants.g * constants.mu_B) 

# Dimensions used in holoviews objects.
d = SimpleNamespace(B=hv.Dimension('$B$', unit='T'), 
                    mu=hv.Dimension('$\mu$', unit='meV'), 
                    gap=hv.Dimension(('gap', r'$E_\mathrm{gap}/\Delta$')),
                    E=hv.Dimension('$E$', unit='meV'),
                    k=hv.Dimension('$k_x$'),
                    xi_inv=hv.Dimension(r'$\xi^-1$', unit=r'nm$^-1$'),
                    xi=hv.Dimension(r'$\xi$', unit=r'nm'))


def make_params(alpha=20, 
                B_x=0, 
                B_y=0,
                B_z=0,
                mu=0,
                mu_sc=0,
                mu_sm=0,
                mu_B=constants.mu_B,
                t=constants.t, 
                g=constants.g,
                orbital=False,
                **kwargs):

    """Function that creates a namespace with parameters.
    Parameters:
    -----------
    alpha : float
        Spin-orbit coupling strength in units of meV*nm.
    B_x, B_y, B_z : float
        The magnetic field strength in the x, y and z direction in units of Tesla.
    Delta : float
        The superconducting gap in units of meV.
    mu : float
        The chemical potential in units of meV.
    mu_sm, mu_sc : float
        The chemical potential in in the SM and SC units of meV.
    mu_B : float
        Bohr magneton in meV/K.
    t : float
        Hopping parameter in meV * nm^2.
    g : float
        Lande g factor.
    orbital : bool
        Switches the orbital effects on and off.
    Returns:
    --------
    p : SimpleNamespace object
        A simple container that is used to store Hamiltonian parameters.
    """

    return SimpleNamespace(t=t, 
                           g=g, 
                           mu_B=mu_B, 
                           alpha=alpha, 
                           B_x=B_x, 
                           B_y=B_y, 
                           B_z=B_z, 
                           mu=mu,
                           mu_sc=mu_sc,
                           mu_sm=mu_sm,
                           orbital=orbital,
                           **kwargs)


def trs(m):
    """Apply time reversal symmetry to a column vector or matrix m.

    The time reversal symmetry is given by the operator i * sigma_y * K, with K
    complex conjugation and sigma_y acting on the spin degree of freedom.

    Parameters:
    -----------
    m : numpy array
        The vector or matrix to which TRS is applied.

    Returns:
    --------
    m_reversed : numpy array
        The vector TRS * m as a NumPy array.

    Notes:
    ------
    Implementation inspired by kwant.rmt.
    """
    permutation = np.arange(m.shape[0])
    sign = 2 * (permutation % 2) - 1
    permutation -= sign
    return sign.reshape(-1, 1) * m.conj()[permutation]


class TRIInfiniteSystem(kwant.builder.InfiniteSystem):
    def __init__(self, lead, trs):
        """A lead with time reversal invariant modes."""
        self.__dict__ = lead.__dict__
        self.trs = trs

    def modes(self, energy=0, args=()):
        prop_modes, stab_modes = \
            super(TRIInfiniteSystem, self).modes(energy=energy, args=args)
        n = stab_modes.nmodes
        stab_modes.vecs[:, n:(2*n)] = self.trs(stab_modes.vecs[:, :n])
        stab_modes.vecslmbdainv[:, n:(2*n)] = \
            self.trs(stab_modes.vecslmbdainv[:, :n])
        prop_modes.wave_functions[:, n:] = \
            self.trs(prop_modes.wave_functions[:, :n])
        return prop_modes, stab_modes


def discretized_hamiltonian(a, dim, holes=False):
    """Discretizes a Hamiltonian.

    Parameters:
    -----------
    a : int
        Lattice constant in nm.
    dim : int
        Dimension of system, 2D or 3D.
    holes : bool
        If False, Hamiltonian will only be in spin-space,
        if True also in particle-hole space (BdG Hamiltonian),
        used for calculating Majorana decay length.
    """
    if dim not in [2, 3]: raise(NotImplementedError)
    k_x, k_y, k_z = momentum_operators
    t, B_x, B_y, B_z, mu_B, mu, mu_sm, mu_sc, alpha, g, V, Delta = sympy.symbols(
        't B_x B_y B_z mu_B mu mu_sm mu_sc alpha g V Delta', real=True)
    k = sympy.sqrt(k_x**2 + k_y**2 + (k_z**2 if dim==3 else 0))

    if not holes:
        ham = ((t * k**2 - mu) * s0 +
                   alpha * (k_y * sx - k_x * sy) +
                   0.5 * g * mu_B * (B_x * sx + B_y * sy + B_z * sz))
    else:
        ham = ((t * k**2 - mu) * kr(s0, sz) +
                   alpha * (k_y * kr(sx, sz) - k_x * kr(sy, sz)) +
                   0.5 * g * mu_B * (B_x * kr(sx, s0) + B_y * kr(sy, s0)) +
                   Delta * kr(s0, sx))
    
    args = dict(lattice_constant=a, discrete_coordinates=set(['x', 'y', 'z'][:dim]))
    tb_sm = Discretizer(ham.subs(mu, mu_sm).subs(Delta, 0), **args)
    tb_sc = Discretizer(ham.subs([(g, 0), (mu, mu_sc), (alpha, 0), (k_x, 0), (k_z, 0)]), **args)
    return tb_sm, tb_sc


def peierls(val, ind, a, c=constants):
    """Peierls substitution, takes hopping functions.
    See usage in NS_infinite_2D_3D()"""
    def phase(s1, s2, p):
        A_site = [0, 0, p.B_x * s1.pos[1]][ind] * a * 1e-18 * c.eV / c.hbar
        return np.exp(-1j * A_site)

    def with_phase(s1, s2, p):
        hop = val(s1, s2, p).astype('complex128')
        phi = phase(s1, s2, p)
        if p.orbital:
            if hop.shape[0] == 2:
                hop *= phi
            elif hop.shape[0] == 4:
                hop *= np.array([phi, phi.conj(), phi, phi.conj()], dtype='complex128')
        return hop

    return with_phase


def NS_infinite_2D_3D(a=10, W=100, H=100, dim=3, normal_lead=False, sc_lead=True, holes=False):
    """Makes a square shaped wire.

    Parameters:
    -----------
    a : int
        Lattice constant in nm.
    W : int
        Width of system in nm.
    H : int
        Height of system in nm (ignored if dim=2).
    dim : int
        Dimension of system, 2D or 3D.
    normal_lead : bool
        Attaches a SM lead to the sytem, used for
        calculating transmission.
    sc_lead : bool
        Attaches a SC lead to the sytem.
    holes : bool
        If False, Hamiltonian will only be in spin-space,
        if True also in particle-hole space, used for calculating
        Majorana decay length.

    Returns:
    --------
    syst : kwant.builder.(In)finiteSystem object
        The finalized (in)finite system.
    """
    tb_sm, tb_sc = discretized_hamiltonian(a, dim, holes)
    lat = tb_sm.lattice
    syst = kwant.Builder(kwant.TranslationalSymmetry((a, 0, 0)[:dim]))
    lead_sc = kwant.Builder(kwant.TranslationalSymmetry((a, 0, 0)[:dim], (0, -a, 0)[:dim]))
    lead_sm = kwant.Builder(kwant.TranslationalSymmetry((a, 0, 0)[:dim], (0, -a, 0)[:dim]))
    
    if dim == 2:
        def shape_func_sm(W, H):
            def shape(pos):
                (x, y) = pos
                return 0 < y <= W
            return (shape, (0, W/2))

        def shape_func_sc(H):
            def shape(pos):
                (x, y) = pos
                return y <= 0
            return (shape, (0, 0))

    elif dim == 3:
        def shape_func_sm(W, H):
            def shape(pos):
                (x, y, z) = pos
                return 0 < y <= W and -H/2 < z <= H/2
            return (shape, (0, W, 0))

        def shape_func_sc(H):
            def shape(pos):
                (x, y, z) = pos
                return y <= 0 and -H/2 < z <= H/2
            return (shape, (0, 0, 0))

    shape_sm = shape_func_sm(W, H)
    shape_sc = shape_func_sc(H)

    syst[lat.shape(*shape_sm)] = tb_sm.onsite
    lead_sc[lat.shape(*shape_sc)] = tb_sc.onsite
    lead_sm[lat.shape(*shape_sc)] = tb_sm.onsite

    for hop, val in tb_sm.hoppings.items():
        ind = np.argmax(hop.delta)
        syst[hop] = peierls(val, ind, a)
        lead_sm[hop] = val

    for hop, val in tb_sc.hoppings.items():
        lead_sc[hop] = val

    syst = wraparound(syst)
    
    if sc_lead:
        syst.attach_lead(wraparound(lead_sc, keep=1))

    if normal_lead:
        syst.attach_lead(wraparound(lead_sm, keep=1).reversed())
    
    fsyst = syst.finalized()
    fsyst.leads = [TRIInfiniteSystem(lead, trs) for lead in fsyst.leads]
    return fsyst


def energy_operator(syst, p, k_x):
    """Returns the operator of Eq. (11) of paper.

    Parameters:
    -----------
    syst : kwant.builder.InfiniteSystem object
        The finalized system.
    p : types.SimpleNamespace object
        A simple container that is used to store Hamiltonian parameters.
    k_x : float
        Momentum for which the energies are calculated.

    Returns:
    --------
    operator : numpy array
        Operator in Eq. (11)."""
    smat_min = kwant.smatrix(syst, args=[p, -k_x]).data
    smat_plus = kwant.smatrix(syst, args=[p, +k_x]).data
    smat_prod = smat_plus.T.conj() @ smat_min.T
    return 0.5 * np.eye(smat_prod.shape[0]) - 0.25 * (smat_prod + smat_prod.T.conj())


def energies_over_delta(syst, p, k_x):
    """Same as energy_operator(), but returns the 
    square-root of the eigenvalues"""
    operator = energy_operator(syst, p, k_x)
    return np.sqrt(np.linalg.eigvalsh(operator))


def find_gap(syst, p, num=201):
    """Find the mimimum in energy in a range of momenta in one third
    of the Brillioun zone.

    Parameters:
    -----------
    syst : kwant.builder.InfiniteSystem object
        The finalized system.
    p : types.SimpleNamespace object
        A simple container that is used to store Hamiltonian parameters.
    num : int
        Number of momenta, more momenta are needed with orbital effects
        in 3D at B_x > 0.5 T.

    Returns:
    --------
    (E, k_x) : tuple of floats
        Tuple of minimum energy found at k_x.
    """
    ks = np.linspace(-0.001, 1, num)
    eigvals = np.array([energies_over_delta(syst, p, k_x) for k_x in ks])
    ind_min, _ = np.unravel_index(eigvals.argmin(), eigvals.shape)
    
    if ind_min == 0:
        bounds = (ks[0], ks[1])
    elif ind_min == num - 1:
        bounds = (ks[num - 2], ks[num-1])
    else:
        bounds = (ks[ind_min - 1], ks[ind_min + 1])
        
    res = minimize_scalar(lambda k_x: energies_over_delta(syst, p, k_x).min(),
                          bounds=bounds, method='bounded')
    k_x = res.x
    E = res.fun
    return E, k_x


def plot_bands(syst, p, ks=None):
    """Plot bandstructure using Eq. (11) of the paper.

    Parameters:
    -----------
    syst : kwant.builder.InfiniteSystem object
        The finalized system.
    p : types.SimpleNamespace object
        A simple container that is used to store Hamiltonian parameters.
    ks : numpy array or None
        Range of momenta for which the energies are calculated.

    Returns:
    --------
    plot : hv.Path object
        Curve of k vs. E_gap/Delta.
    """
    if ks is None:
        ks = np.linspace(-2, 2, 200)
    eigvals = np.array([energies_over_delta(syst, p, k_x) for k_x in ks])
    return hv.Path((ks, eigvals), kdims=[d.k, r'$E/\Delta$'])[:, 0:1.1]


def modes(h_cell, h_hop, tol=1e6):
    """Compute the eigendecomposition of a translation operator of a lead.

    Adapted from kwant.physics.leads.modes such that it returns the eigenvalues.

    Parameters:
    ----------
    h_cell : numpy array, real or complex, shape (N, N) The unit cell
        Hamiltonian of the lead unit cell.
    h_hop : numpy array, real or complex, shape (N, M)
        The hopping matrix from a lead cell to the one on which self-energy
        has to be calculated (and any other hopping in the same direction).
    tol : float
        Numbers and differences are considered zero when they are smaller
        than `tol` times the machine precision.

    Returns
    -------
    ev : numpy array
        Eigenvalues of the translation operator in the form lambda=exp(i*k).
    """
    m = h_hop.shape[1]
    n = h_cell.shape[0]

    if (h_cell.shape[0] != h_cell.shape[1] or
            h_cell.shape[0] != h_hop.shape[0]):
        raise ValueError("Incompatible matrix sizes for h_cell and h_hop.")

    # Note: np.any(h_hop) returns (at least from numpy 1.6.1 - 1.8-devel)
    #       False if h_hop is purely imaginary
    if not (np.any(h_hop.real) or np.any(h_hop.imag)):
        v = np.empty((0, m))
        return (kwant.physics.PropagatingModes(np.empty((0, n)), np.empty((0,)),
                                               np.empty((0,))),
                kwant.physics.StabilizedModes(np.empty((0, 0)),
                                              np.empty((0, 0)), 0, v))

    # Defer most of the calculation to helper routines.
    matrices, v, extract = kwant.physics.leads.setup_linsys(
        h_cell, h_hop, tol, None)
    ev = kwant.physics.leads.unified_eigenproblem(*(matrices + (tol,)))[0]

    return ev


def slowest_evan_mode(syst, p, a, c=constants, return_ev=False):
    """Find the slowest decaying (evanescent) mode.

    It uses an adapted version of the function kwant.physics.leads.modes,
    in such a way that it returns the eigenvalues of the translation operator
    (lamdba = e^ik). The imaginary part of the wavevector k, is the part that
    makes it decay. The inverse of this Im(k) is the size of a Majorana bound
    state. The norm of the eigenvalue that is closest to one is the slowes
    decaying mode. Also called decay length.

    Parameters:
    -----------
    syst : kwant.builder.InfiniteSystem object
        The finalized system.
    p : types.SimpleNamespace object
        A simple container that is used to store Hamiltonian parameters.
    c : types.SimpleNamespace object
        A namespace container with all constant (fundamental) parameters used.

    Returns:
    --------
    majorana_length : float
        The length of the Majorana.
    """
    def H(k_x):
        ham = kwant.solvers.default.hidden_instance._make_linear_sys
        return ham(syst, [0], args=[p, k_x], realspace=True)[0].lhs.todense()

    h = (H(0) + H(np.pi)) / 2
    t_plus_t_ = (H(0) - H(np.pi)) / 2
    t_min_t_ = (H(np.pi/2) - H(-np.pi/2)) / 2j
    t = (t_plus_t_ + t_min_t_) / 2

    ev = modes(h, t)
    norm = ev * ev.conj()
    idx = np.abs(norm - 1).argmin()
    if return_ev:
        return ev[idx]
    majorana_length = np.abs(a / np.log(ev[idx]).real)
    return majorana_length


def plot_phase(dview, lview, p, W, H, dim, num_k=201, fname=None, Bs=None, mus=None, a=10, async_parallel=True):
    """Calculates a phase diagram of bandgap sizes in (B, mu) space in parallel.
    
    Parameters:
    -----------
    dview : DirectView ipyparallel object
        client = ipyparallel.Client(); dview = client[:]
    lview : LoadedBalanced view ipyparallel object
        client = client.load_balanced_view()
    p : shortjunction.SimpleNamespace object
        Container with all parameters for Hamiltonian
    W : int
        Width of system in nm.
    H : int
        Height of system in nm (ignored if dim=2).
    dim : int
        Dimension of system, 2D or 3D.
    num_k : int
        Number of momenta on which the bandstructure is calculated.
    Bs : numpy array or list
        Range of values of magnetic field on which the bandgap is calculated.
    mus : numpy array or list
        Range of values of chemical potentials on which the bandgap is calculated.
    a : int
        Discretization constant in nm.
    async_parallel : bool
        If true it uses lview.map_async, if False it uses a gather scatter formalism,
        which is faster for very short jobs.

    Returns:
    --------
    plot : hv.Image
        Holoviews Image of the phase diagram. The raw data can be accessed
        via plot.data.
    
    Notes:
    ------
    WARNING: This is the opposite behaviour of plot_decay_lengths()
    The parameter `mu_sc` is set to a fixed value, if you want to set it to
    the same values as `mu_sm`, change
    p.update(B_x=x[0], mu_sm=x[1]) ---> p.update(B_x=x[0], mu_sm=x[1], mu_sc=x[1]).
    """
    syst_str = 'syst = NS_infinite_2D_3D(a={}, W={}, H={}, dim={})'.format(a, W, H, dim)
    dview.execute(syst_str, block=True)
    
    dview['p'] = p
    dview['num_k'] = num_k

    if Bs is None:
        Bs = np.linspace(0, 1, 50)
    if mus is None:
        mus = np.linspace(0.1, 15, 50)

    vals = list(product(Bs, mus))
    if async_parallel:
        systs = [ipyparallel.Reference('syst')] * len(vals)
        Es = lview.map_async(lambda x, sys: find_gap(sys, p.update(B_x=x[0], mu_sm=x[1]), num_k), 
                             vals, systs)
        Es.wait_interactive()
        Es = Es.result()
        result = np.array(Es).reshape(len(Bs), len(mus), -1)
    else:
        dview.scatter('xs', vals, block=True)
        dview.execute('Es = [find_gap(syst, p.update(B_x=x[0], mu_sm=x[1]), num_k) for x in xs]',
                      block=True)
        Es = dview.gather('Es', block=True)
        result = np.array(Es).reshape(len(Bs), len(mus), -1)
    
    gaps = result[:, :, 0]
    k_xs = result[:, :, 1]

    bounds = (Bs.min(), mus.min(), Bs.max(), mus.max())

    kwargs = {'kdims': [d.B, d.mu],
              'vdims': [d.gap],
              'bounds': bounds,
              'label': 'Band gap'}

    plot = hv.Image(np.rot90(gaps), **kwargs)
    plot.cdims.update(dict(p=p, k_xs=k_xs, Bs=Bs, mus=mus, W=W, H=H, dim=dim, 
                           constants=constants, num_k=num_k,
                           git_hash=get_git_revision_hash()))
    return plot


def plot_decay_lengths(dview, lview, p, W, H, dim, fname=None, Bs=None, mus=None, a=10, async_parallel=False):
    """Calculates a phase diagram of Majorana decay lengths (nm) 
    in (B, mu) space.
    
    Parameters:
    -----------
    dview : DirectView ipyparallel object
        client = ipyparallel.Client(); dview = client[:]
    lview : LoadedBalanced view ipyparallel object
        client = client.load_balanced_view()
    p : shortjunction.SimpleNamespace object
        Container with all parameters for Hamiltonian
    W : int
        Width of system in nm.
    H : int
        Height of system in nm (ignored if dim=2).
    dim : int
        Dimension of system, 2D or 3D.
    Bs : numpy array or list
        Range of values of magnetic field on which the bandgap is calculated.
    mus : numpy array or list
        Range of values of chemical potentials on which the bandgap is calculated.
    a : int
        Discretization constant in nm.
    async_parallel : bool
        If true it uses lview.map_async, if False it uses a gather scatter formalism,
        which is faster for very short jobs.

    Returns:
    --------
    plot : hv.Image
        Holoviews Image of the phase diagram. The raw data can be accessed
        via plot.data.
    
    Notes:
    ------
    WARNING: This is the opposite behaviour of plot_phase()
    The parameter `mu_sc` is equal to `mu_sm`, if you want to set it to
    a fixed value
    p.update(B_x=x[0], mu_sm=x[1], mu_sc=x[1]) ---> p.update(B_x=x[0], mu_sm=x[1]).
    """
    syst_str = 'syst = NS_infinite_2D_3D(a={}, W={}, H={}, dim={}, holes=True)'.format(a, W, H, dim)
    dview.execute(syst_str, block=True)

    dview['p'] = p
    dview['a'] = a
    
    if Bs is None:
        Bs = np.linspace(0, 2, 50)
    if mus is None:
        mus = np.linspace(0.1, 15, 50)
    
    vals = list(product(Bs, mus))
    if async_parallel:

        systs = [ipyparallel.Reference('syst')] * len(vals)
        decay_lengths = lview.map_async(lambda x, sys:
            slowest_evan_mode(sys, p.update(B_x=x[0], mu_sm=x[1], mu_sc=x[1]), a), vals, systs)

        decay_lengths.wait_interactive()
        result = np.array(decay_lengths.result()).reshape(len(Bs), len(mus))
    else:
        dview.scatter('xs', vals, block=True)
        dview.execute("""decay_lengths = [slowest_evan_mode(syst, 
                      p.update(B_x=x[0], mu_sm=x[1], mu_sc=x[1]), a) for x in xs]""")
        decay_lengths = dview.gather('decay_lengths', block=True)
        result = np.array(decay_lengths).reshape(len(Bs), len(mus))
    
    bounds = (Bs.min(), mus.min(), Bs.max(), mus.max())

    kwargs = {'kdims': [d.B, d.mu],
              'vdims': [d.xi],
              'bounds': bounds,
              'label': 'Decay length'}

    plot = hv.Image(np.rot90(result), **kwargs)
    plot.cdims.update(dict(p=p, Bs=Bs, mus=mus, W=W, H=H, dim=dim, constants=constants,
    	                   git_hash=get_git_revision_hash()))
    return plot


def Ez_to_B(Ez, constants=constants):
    """Converts from Zeeman energy to magnetic field"""
    return 2 * Ez / (constants.g * constants.mu_B)


def B_to_Ez(B, constants=constants):
    """Converts from magnetic field to Zeeman energy"""
    return 0.5 * constants.g * constants.mu_B * B


def sparse_eigs(ham, n_eigs, n_vec_lanczos, sigma=0):
    """Compute eigenenergies using MUMPS as a sparse solver.

    Parameters:
    ----------
    ham : coo_matrix
        The Hamiltonian of the system in sparse representation..
    n_eigs : int
        The number of energy eigenvalues to be returned.
    n_vec_lanczos : int
        Number of Lanczos vectors used by the sparse solver.
    sigma : float
        Parameter used by the shift-inverted method. See
        documentation of scipy.sparse.linalg.eig

    Returns:
    --------
    A list containing the sorted energy levels. Only positive
    energies are returned.
    """
    class LuInv(sla.LinearOperator):
        def __init__(self, A):
            inst = kwant.linalg.mumps.MUMPSContext()
            inst.factor(A, ordering='metis')
            self.solve = inst.solve
            try:
                super(LuInv, self).__init__(shape=A.shape, dtype=A.dtype,
                                            matvec=self._matvec)
            except TypeError:
                super(LuInv, self).__init__(shape=A.shape, dtype=A.dtype)

        def _matvec(self, x):
            return self.solve(x.astype(self.dtype))

    ev, evecs = sla.eigs(ham, k=n_eigs,
                         OPinv=LuInv(ham), sigma=sigma, ncv=n_vec_lanczos)

    energies = list(ev.real)
    return energies


def NS_finite(a, L, W, W_sc):
    """Makes two-dimensional NS junction, normal part
    of width W connected to the superconducting part of
    width W_sc.

    Parameters:
    -----------
    a: int
        Lattice constant in units of nm.
    L: int
        Length of the wire and superconductor in x-direction.
    W: int
        Width of the normal section of the wire in units of a.
    W_sc: int
        Width of the superconducting section of the wire in units of a.

    Returns:
    --------
    syst: kwant.builder.FiniteSystem object
        Finalized tight-binding systtem.
    """
    lat = kwant.lattice.square(a)
    syst = kwant.Builder()

    def onsite_sm(site, p):
        return (4 * p.t / a**2 - p.mu) * np.kron(s0, sz) + p.Ez * np.kron(sx, s0)

    def hopx_sm(site1, site2, p):
        return -p.t / a**2 * np.kron(s0, sz) - 0.5j * p.alpha / a * np.kron(sy, sz)

    def hopy_sm(site1, site2, p):
        return -p.t / a**2 * np.kron(s0, sz) + 0.5j * p.alpha / a * np.kron(sx, sz)

    def onsite_sc(site, p):
        return ((2 * p.t / a**2 + 2 * p.tpar / a**2 - p.mu) * np.kron(s0, sz) +
                p.delta * np.kron(s0, sx))

    def hopx_sc(site1, site2, p):
        return -p.tpar / a**2 * np.kron(s0, sz)

    def hopy_sc(site1, site2, p):
        return -p.t / a**2 * np.kron(s0, sz)

    # Onsite energies
    syst[(lat(i, j) for i in range(L) for j in range(W+1))] = onsite_sm
    syst[(lat(i, j) for i in range(L) for j in range(W+1, W+W_sc))] = onsite_sc
    
    # Hopping energies
    syst[((lat(i, j), lat(i+1, j)) for i in range(L-1) for j in range(W+1))] = hopx_sm
    syst[((lat(i, j), lat(i+1, j)) for i in range(L-1) for j in range(W+1, W+W_sc))] = hopx_sc
    syst[((lat(i, j), lat(i, j+1)) for i in range(L) for j in range(W))] = hopy_sm
    syst[((lat(i, j), lat(i, j+1)) for i in range(L) for j in range(W, W+W_sc-1))] = hopy_sc

    return syst.finalized()


def NS_infinite(a, L):
    """Makes two-dimensional NS junction, with a 2D semi-infinite
    superconducting lead connected to the normal part with finite
    length and infinite in the direction parallel to the interface.

    Parameters:
    -----------
    a : int
        Lattice constant in units of nm.
    L : int
        Width of the normal parts in units of nm.

    Returns:
    --------
    syst: kwant.builder.FiniteSystem object
        Finalized tight-binding systtem.
    """

    sx, sy, sz = [np.array(sympy.physics.matrices.msigma(i)).astype(np.complex)
                  for i in range(1, 4)]
    s0 = np.eye(2)
    lat = kwant.lattice.square(a)

    def onsite(site, p):
        return ((4 * p.t / a**2 - p.mu) * s0 + p.Ez * sx)

    def hopx(site1, site2, p):
        return -p.t / a**2 * s0 - 0.5j * p.alpha / a * sy
    
    def hopy(site1, site2, p):
        return -p.t / a**2 * s0 + 0.5j * p.alpha / a * sx

    def lead_onsite(site, p):
        return (2 * p.t / a**2 + 2 * p.tpar / a**2 - p.mu) * s0

    def lead_hopx(site1, site2, p):
        return -p.tpar / a**2 * s0

    def lead_hopy(site1, site2, p):
        return -p.t / a**2 * s0
    
    def shape_sm(pos):
        (x, y) = pos
        return 0 < y <= L
    
    def shape_sc(pos):
        (x, y) = pos
        return y >= 0
    
    # SM part
    sym_sm = kwant.TranslationalSymmetry((a, 0))
    syst = kwant.Builder(sym_sm)
    syst[lat.shape(shape_sm, (0, L / 2))] = onsite
    syst[kwant.HoppingKind((1, 0), lat)] = hopx
    syst[kwant.HoppingKind((0, 1), lat)] = hopy
    
    # SC lead
    lead_sym = kwant.TranslationalSymmetry((a, 0), (0, a))
    lead = kwant.Builder(lead_sym)
    lead[lat.shape(shape_sc, (0, 0))] = lead_onsite
    lead[kwant.HoppingKind((1, 0), lat)] = lead_hopx
    lead[kwant.HoppingKind((0, 1), lat)] = lead_hopy

    syst = wraparound(syst)
    syst.attach_lead(wraparound(lead, keep=1))
    syst = syst.finalized()
    syst.leads = [TRIInfiniteSystem(lead, trs) for lead in syst.leads]
    return syst


def SNS_infinite(a, L, W_sm):
    """Makes two-dimensional SNS junction with orbital effect in N part

    Parameters:
    -----------
    a : int
        Lattice constant in units of nm.
    L : int
        Width of the superconducting in units of sites.
    W_sm : int
        Width of the normal (semi-conducting) parts in units of sites.

    Returns:
    --------
    syst: kwant.builder.FiniteSystem object
        Finalized tight-binding systtem.
    """
    sx, sy, sz = [np.array(sympy.physics.matrices.msigma(i)).astype(np.complex)
                  for i in range(1, 4)]
    s0 = np.eye(2)
    lat = kwant.lattice.square(a)

    def onsite_sm(site, p):
        return (4 * p.t / a**2 - p.mu) * np.kron(s0, sz) + B_to_Ez(p.B) * np.kron(sx, s0)

    def onsite_sc(site, p):
        return ((2 * p.t / a**2 + 2 * p.tpar / a**2 - p.mu) * np.kron(s0, sz) +
                p.delta * np.kron(s0, sx))

    def hopx_sm(site1, site2, p):
        y1 = site1.tag[0]
        y2 = site2.tag[0]
        phi = 0.25 * np.pi * a**2 * p.D**2 * constants.eV * 1e-18 * p.B / constants.hbar / W_sm
        exp_phi = expm(1j * phi * np.kron(s0, sz))
        return (-p.t / a**2 * exp_phi @ np.kron(s0, sz) +
                       0.5j * exp_phi @ np.kron(sx, sz) * np.sin((y1 + y2) / p.D) * p.alpha / a)

    def hopy_sm(site1, site2, p):
        return -p.t / a**2 * np.kron(s0, sz) - 0.5j * p.alpha / a * np.kron(sy, sz)    

    def hopx_sc(site1, site2, p):
        return -p.tpar / a**2 * np.kron(s0, sz)

    def hopy_sc(site1, site2, p):
        return -p.t / a**2 * np.kron(s0, sz)

    sym = kwant.TranslationalSymmetry((0, a))
    syst = kwant.Builder(sym)

    syst[(lat(i, j) for i in range(L) for j in range(2))] = onsite_sc
    syst[(lat(i, j) for i in range(L, L+W_sm) for j in range(2))] = onsite_sm
    syst[(lat(i, j) for i in range(L+W_sm, 2*L+W_sm) for j in range(2))] = onsite_sc

    syst[((lat(i, j), lat(i+1, j)) for i in range(L) for j in range(2))] = hopx_sc
    syst[((lat(i, j), lat(i+1, j)) for i in range(L, L+W_sm) for j in range(2))] = hopx_sm
    syst[((lat(i, j), lat(i+1, j)) for i in range(L+W_sm, 2*L+W_sm-1) for j in range(2))] = hopx_sc

    syst[((lat(i, 0), lat(i, 1)) for i in range(L))] = hopy_sc
    syst[((lat(i, 0), lat(i, 1)) for i in range(L, L+W_sm))] = hopy_sm
    syst[((lat(i, 0), lat(i, 1)) for i in range(L+W_sm, 2*L+W_sm-1))] = hopy_sc
    
    syst = wraparound(syst)
    return syst.finalized()
