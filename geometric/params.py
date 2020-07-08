"""
params.py: Optimization parameters and user options

Copyright 2016-2020 Regents of the University of California and the Authors

Authors: Lee-Ping Wang, Chenchen Song

Contributors: Yudong Qiu, Daniel G. A. Smith, Alberto Gobbi, Josh Horton

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import division
import argparse
import numpy as np
from .nifty import logger

class OptParams(object):
    """
    Container for optimization parameters.
    The parameters used to be contained in the command-line "args",
    but this was dropped in order to call Optimize() from another script.
    """
    def __init__(self, **kwargs):
        # Whether we are optimizing for a transition state. This changes a number of default parameters.
        self.transition = kwargs.get('transition', False)
        # Handle convergence criteria; this edits the kwargs
        self.convergence_criteria(**kwargs)
        # Threshold (in a.u. / rad) for activating alternative algorithm that enforces precise constraint satisfaction
        self.enforce = kwargs.get('enforce', 0.0)
        # Small eigenvalue threshold
        self.epsilon = kwargs.get('epsilon', 1e-5)
        # Interval for checking the coordinate system for changes
        self.check = kwargs.get('check', 0)
        # More verbose printout
        self.verbose = kwargs.get('verbose', False)
        # Rational function optimization (experimental)
        self.rfo = kwargs.get('rfo', False)
        # Starting value of the trust radius
        self.trust = kwargs.get('trust', 0.1)
        # Maximum value of trust radius
        self.tmax = kwargs.get('tmax', 0.3)
        self.trust = min(self.tmax, self.trust)
        # Maximum number of optimization cycles
        self.maxiter = kwargs.get('maxiter', 300)
        # Use updated constraint algorithm implemented 2019-03-20
        self.conmethod = kwargs.get('conmethod', 0)
        # Write Hessian matrix at optimized structure to text file
        self.write_cart_hess = kwargs.get('write_cart_hess', None)
        # CI optimizations sometimes require tiny steps
        self.meci = kwargs.get('meci', False)
        # Output .xyz file name may be set separately in
        # run_optimizer() prior to calling Optimize().
        self.xyzout = kwargs.get('xyzout', None)
        # Name of the qdata.txt file to be written.
        # The CLI is designed so the user passes true/false instead of the file name.
        self.qdata = 'qdata.txt' if kwargs.get('qdata', False) else None
        # Whether to calculate or read a Hessian matrix.
        self.hessian = kwargs.get('hessian', None)
        if self.hessian is None:
            # Default is to calculate Hessian in the first step if searching for a transition state.
            # Otherwise the default is to never calculate the Hessian.
            if self.transition: self.hessian = 'first'
            else: self.hessian = 'never'
        if self.hessian.startswith('file:'):
            if os.path.exists(self.hessian[5:]):
                # If a path is provided for reading a Hessian file, read it now.
                self.hess_data = np.loadtxt(self.hessian[5:])
            else:
                raise IOError("No Hessian data file found at %s" % self.hessian)
        elif self.hessian.lower() in ['never', 'first', 'each', 'exit']:
            self.hessian = self.hessian.lower()
        else:
            raise RuntimeError("Hessian command line argument can only be never, first, each, exit, or file:<path>")
        # Reset Hessian to guess whenever eigenvalues drop below epsilon
        self.reset = kwargs.get('reset', None)
        if self.reset is None: self.reset = not (self.transition or self.hessian == 'each')

    def convergence_criteria(self, **kwargs):
        criteria = kwargs.get('converge', [])
        if len(criteria)%2 != 0:
            raise RuntimeError('Please pass an even number of options to --converge')
        for i in range(int(len(criteria)/2)):
            key = 'convergence_' + criteria[2*i].lower()
            try:
                val = float(criteria[2*i+1])
                logger.info('Using convergence criteria: %s %.2e\n' % (key, val))
            except ValueError:
                # This must be a set
                val = str(criteria[2*i+1])
                logger.info('Using convergence criteria set: %s %s\n' % (key, val))
            kwargs[key] = val
        # convergence dictionary to store criteria stored in order of energy, grms, gmax, drms, dmax
        # 'GAU' contains the default convergence criteria that are used when nothing is passed.
        convergence_sets = {'GAU': [1e-6, 3e-4, 4.5e-4, 1.2e-3, 1.8e-3],
                            'NWCHEM_LOOSE': [1e-6, 3e-3, 4.5e-3, 3.6e-3, 5.4e-3],
                            'GAU_LOOSE': [1e-6, 1.7e-3, 2.5e-3, 6.7e-3, 1e-2],
                            'TURBOMOLE': [1e-6, 5e-4, 1e-3, 5.0e-4, 1e-3],
                            'INTERFRAG_TIGHT': [1e-6, 1e-5, 1.5e-5, 4.0e-4, 6.0e-4],
                            'GAU_TIGHT': [1e-6, 1e-5, 1.5e-5, 4e-5, 6e-5],
                            'GAU_VERYTIGHT': [1e-6, 1e-6, 2e-6, 4e-6, 6e-6]}
        # Q-Chem style convergence criteria (i.e. gradient and either energy or displacement)
        self.qccnv = kwargs.get('qccnv', False)
        # Molpro style convergence criteria (i.e. gradient and either energy or displacement, with different defaults)
        self.molcnv = kwargs.get('molcnv', False)
        # Check if there is a convergence set passed else use the default
        set_name = kwargs.get('convergence_set', 'GAU').upper()
        # If we have extra keywords apply them here else use the set
        # Convergence criteria in a.u. and Angstrom
        self.Convergence_energy = kwargs.get('convergence_energy', convergence_sets[set_name][0])
        self.Convergence_grms = kwargs.get('convergence_grms', convergence_sets[set_name][1])
        self.Convergence_gmax = kwargs.get('convergence_gmax', convergence_sets[set_name][2])
        self.Convergence_drms = kwargs.get('convergence_drms', convergence_sets[set_name][3])
        self.Convergence_dmax = kwargs.get('convergence_dmax', convergence_sets[set_name][4])
        # Convergence criteria that are only used if molconv is set to True
        self.Convergence_molpro_gmax = kwargs.get('convergence_molpro_gmax', 3e-4)
        self.Convergence_molpro_dmax = kwargs.get('convergence_molpro_dmax', 1.2e-3)

    def printInfo(self):
        if self.transition:
            logger.info(' Transition state optimization requested.\n')
        if self.hessian == 'first':
            logger.info(' Hessian will be computed on the first step.\n')
        elif self.hessian == 'each':
            logger.info(' Hessian will be computed for each step.\n')
        elif self.hessian == 'exit':
            logger.info(' Hessian will be computed for first step, then program will exit.\n')
        elif self.hessian.startswith('file:'):
            logger.info(' Hessian data will be read from file: %s\n' % self.hessian[5:])

def str2bool(v):
    """ Allows command line options such as "yes" and "True" to be converted into Booleans. """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(*args):
    
    """ Read user input. Designed to be called by optimize.main() passing in sys.argv[1:] """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--coordsys', type=str, default='tric', help='Coordinate system: "cart" for Cartesian, "prim" for Primitive (a.k.a redundant), '
                        '"dlc" for Delocalized Internal Coordinates, "hdlc" for Hybrid Delocalized Internal Coordinates, "tric" for Translation-Rotation'
                        'Internal Coordinates (default).')
    parser.add_argument('--engine', type=str, default='tera', help='Specify engine for computing energies and gradients, '
                        '"tera" for TeraChem (default), "qchem" for Q-Chem, "psi4" for Psi4, "openmm" for OpenMM, "gmx" for Gromacs, "molpro" for Molpro.')
    parser.add_argument('--meci', type=str, default=None, help='Provide second input file and search for minimum-energy conical '
                        'intersection or crossing point between two SCF solutions (TeraChem and Q-Chem supported).'
                        'Or, provide "engine" if the engine directly provides the MECI objective function and gradient.')
    parser.add_argument('--meci_sigma', type=float, default=3.5, help='Sigma parameter for MECI optimization;'
                        'only used if geomeTRIC computes the MECI objective function from 2 energies/gradients.')
    parser.add_argument('--meci_alpha', type=float, default=0.025, help='Alpha parameter for MECI optimization;'
                        'only used if geomeTRIC computes the MECI objective function from 2 energies/gradients.')
    parser.add_argument('--molproexe', type=str, default=None, help='Specify absolute path of Molpro executable.')
    parser.add_argument('--molcnv', action='store_true', help='Use Molpro style convergence criteria instead of the default.')
    parser.add_argument('--prefix', type=str, default=None, help='Specify a prefix for log file and temporary directory.')
    parser.add_argument('--displace', action='store_true', help='Write out the displacements of the coordinates.')
    parser.add_argument('--fdcheck', action='store_true', help='Check internal coordinate gradients using finite difference.')
    parser.add_argument('--get_hessian', action='store_true', help='Compute Hessian as the numerical Jacobian of the Gradient.')
    parser.add_argument('--enforce', type=float, default=0.0, help='Enforce exact constraints when within provided tolerance (in a.u. and radian)')
    parser.add_argument('--conmethod', type=int, default=0, help='Set to 1 to enable updated constraint algorithm.')
    parser.add_argument('--write_cart_hess', type=str, default=None, help='Convert current Hessian matrix at optimized geometry to Cartesian coords and write to specified file.')
    parser.add_argument('--epsilon', type=float, default=1e-5, help='Small eigenvalue threshold.')
    parser.add_argument('--check', type=int, default=0, help='Check coordinates every N steps to see whether it has changed.')
    parser.add_argument('--verbose', type=int, default=0, help='Set to positive for more verbose printout. 1 = Basic info about optimization step. 2 = Include microiterations.'
                        '3 = Lots of printout from low-level functions.')
    parser.add_argument('--logINI',  type=str, dest='logIni', help='ini file for logging')
    parser.add_argument('--reset', type=str2bool, help='Reset Hessian when eigenvalues are under epsilon. Defaults to True for minimization and False for transition states.')
    parser.add_argument('--transition', action='store_true', help='Search for a first order saddle point / transition state.')
    parser.add_argument('--hessian', type=str, help='Specify when to calculate Cartesian Hessian from finite difference of gradient. '
                        '"never" : Do not calculate or read Hessian data. file:<path> : Read Hessian data in NumPy format from path, e.g. file:run.tmp/hessian/hessian.txt .'
                        '"first" : Calculate or read for the initial structure. "each" : Calculate for each step in the optimization (costly).'
                        '"exit" : Calculate Hessian and then exit without optimizing. Default is "never" for minimization and "initial" for transition state.')
    parser.add_argument('--rfo', action='store_true', help='Use rational function optimization (default is trust-radius Newton Raphson).')
    parser.add_argument('--trust', type=float, default=0.1, help='Starting trust radius.')
    parser.add_argument('--tmax', type=float, default=0.3, help='Maximum trust radius.')
    parser.add_argument('--maxiter', type=int, default=300, help='Maximum number of optimization steps.')
    parser.add_argument('--radii', type=str, nargs="+", default=["Na","0.0"], help='List of atomic radii for coordinate system.')
    parser.add_argument('--pdb', type=str, help='Provide a PDB file name with coordinates and resids to split the molecule.')
    parser.add_argument('--coords', type=str, help='Provide coordinates to override the TeraChem input file / PDB file. The LAST frame will be used.')
    parser.add_argument('--frag', action='store_true', help='Fragment the internal coordinate system by deleting bonds between residues.')
    parser.add_argument('--qcdir', type=str, help='Provide an initial Q-Chem scratch folder e.g. supplied initial guess).')
    parser.add_argument('--qccnv', action='store_true', help='Use Q-Chem style convergence criteria instead of the default.')
    parser.add_argument('--qdata', action='store_true', help='Write qdata.txt containing coordinates, energies, gradients for each structure in optimization.')
    parser.add_argument('--converge', type=str, nargs="+", default=[], help='Custom convergence criteria as key/value pairs.'
                        'Provide the name of a criteria set as "set GAU_LOOSE" or "set TURBOMOLE", and/or set specific criteria using "energy 1e-5" or "grms 1e-3')
    parser.add_argument('--nt', type=int, help='Specify number of threads for running in parallel (for TeraChem this should be number of GPUs)')
    parser.add_argument('--proxy', type=str, default='', help='Specify IP address and the port number for the daemon optimizer in Orquestra')
    parser.add_argument('--delta', type=float, default=1e-2, help='Geometry displacement for numerical derivatives')
    parser.add_argument('--fdorder', type=int, default=4, help='Order of finite-difference approximation to use (2,4 or 6)')
    parser.add_argument('input', type=str, help='TeraChem or Q-Chem input file')
    parser.add_argument('constraints', type=str, nargs='?', help='Constraint input file (optional)')
    args = parser.parse_args(*args)
    return args
