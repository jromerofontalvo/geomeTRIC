import shutil
from collections import OrderedDict
from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np
import re
import os

from .molecule import Molecule
from .engine import Engine
from .nifty import bak, eqcgmx, fqcgmx, bohr2ang, logger, getWorkQueue, queue_up_src_dest, splitall
from .errors import EngineError, ZmachineEngineError

from zmachine.optimizer.proxy import Client
from zmachine.domain.chem import save_molecular_geometry, MolecularGeometry
from zmachine.core.utils import load_value_estimate
import time
import io
import json

class Zmachine(Engine):
    """
    Run a prototypical Zmachine energy and gradient calculation.
    """
    def __init__(self, molecule=None, fdorder=4, d=1e-2, proxy=''):
        self.basis = np.eye(3)
        # format num_points : ([c_{+num_points/2}, ..., c_{1}], denom)
        self.fd_formulae = {2 : ([1.0], 2.0), 4 : ([-1.0, 8.0], 12.0), 6 : ([1.0, -9.0, 45.0], 60.0) }
        # molecule.py can not parse psi4 input yet, so we use self.load_psi4_input() as a walk around
        if molecule is None:
            # create a fake molecule
            molecule = Molecule()
            molecule.elem = ['H']
            molecule.xyzs = [[[0,0,0]]]

        self.fd_options = {}
        self.fd_options['npoint'] = fdorder
        self.fd_options['d'] = d
        print("fd_options: ", self.fd_options)
        if proxy == '':
            self.client = None
        else:
            self.client = Client(proxy, "8080")

        super(Zmachine, self).__init__(molecule)

    def load_zmachine_input(self, zmachinein):
        """ Parse a JSON input file """
        import json
        coords = []
        elems = []
        with open(zmachinein, 'r') as f:
            s = f.read()
            input_dict = json.loads(s)
        self.M = Molecule()
        self.M.elem = input_dict['atoms']['elements']['symbols']
        self.M.xyzs = [np.array(input_dict['atoms']['coords']['3d'], dtype=np.float64).reshape(-1, 3)]
        self.psi4_options = {}
        self.psi4_options['basis'] = input_dict.get('basis', 'sto-3g')
        self.psi4_options['scf_type'] = input_dict.get('scf_type', 'pk')
        self.psi4_options['e_convergence'] = input_dict.get('e_convergence', 11)
        self.psi4_options['d_convergence'] = input_dict.get('d_convergence', 11)

        #print("psi4_options: ", self.psi4_options)
        #print("fd_options: ", self.fd_options)
        #print(self.M.elem)
        #print(self.M.xyzs)

    def compute_energy(self, geom, dirname):
        """ geom is [[np.array, ...]] 
        """
        #create a str representations of molecular geometries and run Psi4
        energies = []
        for at_block in geom:
            at_block_energy = []
            for g in at_block:
                at_block_energy.append(self.request_energy(g, dirname))
            energies.append(at_block_energy)

        return energies

    def request_energy(self, g, dirname=None):
        if self.client == None:
            """ Prepare the input and run Psi4 """
            import psi4
            if not os.path.exists(dirname): os.makedirs(dirname)
            psi4out = os.path.join(dirname, 'output.dat')
            psi4.core.set_output_file(psi4out, True)
            psi4.set_options(self.psi4_options)
            g_str = ''
            for atom, coords in zip(self.M.elem, g):
                g_str += "{0} {1:13.6f} {2:13.6} {3:13.6}\n".format(atom, coords[0], coords[1], coords[2])
            tmp_mol = psi4.geometry(g_str)
            return psi4.energy('scf', mol=tmp_mol)
        else:
            """ Send a request to the proxy """
            # Encode params to json string
            # Create an geometry string in XYZ format
            g_xyz = "{}\n\n".format(len(self.M.elem))
            for atom, coords in zip(self.M.elem, g):
                g_xyz += "{0} {1:15.8f} {2:15.8} {3:15.8}\n".format(atom, coords[0], coords[1], coords[2])

            tmp_mol = MolecularGeometry.from_xyz(g_xyz)
            save_molecular_geometry(tmp_mol, 'current_molecular_geometry.json')
            with open('current_molecular_geometry.json', 'r') as f:
                current_geom_string = f.read()

            # POST params to proxy
            evaluation_id = self.client.post_argument_values(current_geom_string)

            # POST status to EVALUATING
            self.client.post_status("EVALUATING")

            # WAIT for status to be OPTIMIZING
            while self.client.get_status() != "OPTIMIZING":
                time.sleep(1)

            # GET cost function evaluation from proxy
            evaluation_string = self.client.get_evaluation_result(evaluation_id)
            res = json.loads(evaluation_string)

            #For psi4 (will have to write adapters for other templates)
            return res['energy']

            #value_estimate = load_value_estimate(io.StringIO(evaluation_string))
            #return value_estimate.value

    def compute_gradient(self, en):
        assert self.fd_options['npoint'] % 2 == 0 and self.fd_options['npoint'] in self.fd_formulae
        npoint = self.fd_options['npoint']
        en = np.array(en)
        en = en.reshape((-1, 3, npoint))
        disp = self.fd_options['d']
        # apply fd stencil to calculate derivatives
        num = list(self.fd_formulae[npoint][0])
        num.extend([-i for i in reversed(num)])
        denom = self.fd_formulae[npoint][1] * disp
        stencil = np.array(num) / denom
        return np.einsum("ijk,k->ij", en, stencil)

    def calc_new(self, coords, dirname):
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang # in angstrom!!
        geometries = []
        npoint_one_way = self.fd_options['npoint'] // 2
        disp  = self.fd_options['d']
        for at in range(len(self.M.elem)):
            at_disp = []
            for c in range(3):
                for sc in range(npoint_one_way, -npoint_one_way-1, -1):
                    if sc == 0:
                        continue
                    else:
                        g = np.copy(self.M.xyzs[0])
                        g[at] += sc * disp * self.basis[c]
                        at_disp.append(g)
            geometries.append(at_disp)

        energies = self.compute_energy(geometries,dirname)
        ref_gradient = self.compute_gradient(energies)
        ref_energy = self.compute_energy([[self.M.xyzs[0]]], dirname)[0][0] # ugly...

        print("ref_gradient: ", ref_gradient)

        return {'energy':ref_energy, 'gradient':ref_gradient.ravel()}

class Zmachine_batch(Engine):
    """
    Run a prototypical Zmachine energy and gradient calculation.
    """
    def __init__(self, proxy, molecule=None):
        # molecule.py can not parse psi4 input yet, so we use self.load_psi4_input() as a walk around
        if molecule is None:
            # create a fake molecule
            molecule = Molecule()
            molecule.elem = ['H']
            molecule.xyzs = [[[0,0,0]]]

        self.client = Client(proxy, "8080")
        super(Zmachine, self).__init__(molecule)

    def load_zmachine_input(self, zmachinein):
        """ Parse a JSON input file """
        import json
        coords = []
        elems = []
        with open(zmachinein, 'r') as f:
            s = f.read()
            input_dict = json.loads(s)
        self.M = Molecule()
        self.M.elem = input_dict['atoms']['elements']['symbols']
        self.M.xyzs = [np.array(input_dict['atoms']['coords']['3d'], dtype=np.float64).reshape(-1, 3)]

    def calc_new(self, coords, dirname):
        """ Send a request to the proxy """
        # Convert coordinates back to the xyz file
        self.M.xyzs[0] = coords.reshape(-1, 3) * bohr2ang # in angstrom!!
        # Encode params to json string
        save_molecular_geometry(tmp_mol, 'current_molecular_geometry.json')
        with open('current_molecular_geometry.json', 'r') as f:
            current_geom_string = f.read()

        # POST params to proxy
        evaluation_id = self.client.post_argument_values(current_geom_string)

        # POST status to EVALUATING
        self.client.post_status("EVALUATING")

        # WAIT for status to be OPTIMIZING
        while self.client.get_status() != "OPTIMIZING":
            time.sleep(1)

        # GET cost function evaluation from proxy
        evaluation_string = self.client.get_evaluation_result(evaluation_id)
        res = json.loads(evaluation_string) # res is a dict with `energy` : float and `gradient` : list of floats

        return {'energy': res['energy'], 'gradient': np.array(res['gradient'])}
