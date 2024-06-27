from AnalysisG.Templates import SelectionTemplate
import pyc
from AnalysisG.Events import Event 
import torch
from neutrino_reconstruction.nusol import NuSol, DoubleNu, UnitCircle
from neutrino_reconstruction.common import EventNutest, ParticleNutest
from sympy import symbols, cos, sin
from sympy.plotting import plot3d_parametric_line, plot3d_parametric_surface, plot_parametric
import matplotlib.cm as cm
import numpy as np
import scipy.optimize

def make_input(particle):
    return [particle.pt, particle.eta, particle.phi, particle.e]

mT = 172.5 #GeV
mW = 80.379 #GeV

def find_dist(vec, part):
    if vec is None or part is None:
        return None
    return ((float(vec[0]) - part.px)**2 + (float(vec[1]) - part.py)**2 + (float(vec[2]) - part.pz)**2)**0.5

class Selection(SelectionTemplate):
    def __init__(self):
        SelectionTemplate.__init__(self)
        self.nsolutions = {'met_nu+mass_obj' : {i : 0 for i in range(10)},
                           'met_nu+mass_real' : {i : 0 for i in range(10)},
                           'met_reco+mass_obj' : {i : 0 for i in range(10)},
                           'met_reco+mass_real': {i : 0 for i in range(10)}}
        self.met_diff = []
        self.met_phi_diff = []
        self.both_found = False
        self.top_mass = {'solution_found' : [], 'solution_not_found' : []}
        self.top_mass_difference = {'solution_found' : [], 'solution_not_found' : []}
        self.W_mass = {'solution_found' : [], 'solution_not_found' : []}
        self.W_mass_difference = {'solution_found' : [], 'solution_not_found' : []}
        self.met = {'solution_found' : [], 'solution_not_found' : []}
        self.met_phi = {'solution_found' : [], 'solution_not_found' : []}
        self.leps = []
        self.lep_dR = []
        self.nu_dR = []
        self.b_dR = []
        self.blep_dR = []
        self.nulep_dR = []
        self.b_pt = []
        self.lep_pt = []
        self.nu_pt = []


    def Selection(self, event):
        leptons = [child for child in event.TopChildren if child.is_lep]
        if len(leptons) < 2:
            return False
        # if len([1 for child in leptons if abs(child.pdgid) == 15]):
        #     return False
        return True
    
    def ReconstructNu(self, event, plot=False, filename='testfig.png'):
        leptons = [child for child in event.TopChildren if child.is_lep]
        leptons = sorted(leptons, key=lambda x: x.pt, reverse=True)
        l1, l2 = leptons[:2]
        t1 = l1.Parent[0]
        t2 = l2.Parent[0]
        b1 = [child for child in t1.Children if child.is_b][0]
        b2 = [child for child in t2.Children if child.is_b][0]

        l1_nusol = ParticleNutest(l1.pt, l1.eta, l1.phi, l1.e)
        l2_nusol = ParticleNutest(l2.pt, l2.eta, l2.phi, l2.e)
        b1_nusol = ParticleNutest(b1.pt, b1.eta, b1.phi, b1.e)
        b2_nusol = ParticleNutest(b2.pt, b2.eta, b2.phi, b2.e)
        nu1 = [child for child in t1.Children if child.is_nu][0]
        nu2 = [child for child in t2.Children if child.is_nu][0]
        met0 = (nu1 + nu2).pt
        met_phi0 = (nu1 + nu2).phi
        ev = EventNutest(met0, met_phi0)
        mT1 = (l1 + b1 + nu1).Mass
        mW1 = (l1 + nu1).Mass
        sol1 = NuSol(b1_nusol.vec, l1_nusol.vec, mT2=mT1**2, mW2=mW1**2)
        bm1 = sol1.H
        mT2 = (l2 + b2 + nu2).Mass
        mW2 = (l2 + nu2).Mass
        sol2 = NuSol(b2_nusol.vec, l2_nusol.vec, mT2=mT2**2, mW2=mW2**2)
        bm2 = sol2.H
        # print('BaseMatrix 1', sol1.BaseMatrix)
        # print('top mass', mT1)
        # print('W mass', mW1)
        # print('BaseMAtrix 2', sol2.BaseMatrix)
        # print('top mass', mT2)
        # print('W mass', mW2)
        # print(sol2.Z2)
        
        dn = DoubleNu([b1_nusol.vec, b2_nusol.vec], [l1_nusol.vec, l2_nusol.vec], ev, mW1, mT1, mW2, mT2)
        # print(n1)
        # print(n2)
        if plot:
            fn = filename.split('.')
            t = symbols('t')
            alpha = np.array([[cos(t)], [sin(t)], [1]])
            el_nu = dn.solutionSets[0].H_perp.dot(alpha)
            el_met_nu = dn.S.dot(el_nu)
            el_anu = dn.solutionSets[1].H_perp.dot(alpha)
            el_met_anu = dn.S.dot(el_anu)
            p = plot_parametric((el_nu[0][0], el_nu[1][0], (t, -5, 5)),
                                (el_anu[0][0], el_anu[1][0], (t, -5, 5)),
                                (el_met_nu[0][0], el_met_nu[1][0], (t, -5, 5)),
                                (el_met_anu[0][0], el_met_anu[1][0], (t, -5, 5)),
                                legend=False,
                                show=False,
                                dvioptions=['-D','6000'])
            # print(dn.nunu_s)
            p.save(f'{fn[0]}_real_mass.{fn[1]}')
        sol1 = NuSol(b1_nusol.vec, l1_nusol.vec)
        sol2 = NuSol(b2_nusol.vec, l2_nusol.vec)
        V0 = np.outer([ev.vec.px, ev.vec.py, 0], [0, 0, 1])
        S = V0 - UnitCircle()
        H1 = sol1.H_perp
        H2 = S.dot(sol2.H_perp)
        min_dist = None 
        def calculate_distance(phis):
            t1 = H1.dot([[np.cos(phis[0])], [np.sin(phis[0])], [1]])
            t2 = H2.dot([[np.cos(phis[1])], [np.sin(phis[1])], [1]])
            return ((t2[0][0] - t1[0][0])**2 + (t2[1][0] - t1[1][0])**2 + (t2[2][0] - t1[2][0])**2)**0.5
        phi0s = sum([[[p1, p2] for p1 in [0, 3.14/2, -3.14/2, 3.14, -3.14]] for p2 in [0, 3.14/2, -3.14/2, 3.14, -3.14]], start=[])
        phi0s = [[0, 0]]
        for phi0 in phi0s:
            phi_min = scipy.optimize.minimize(calculate_distance, x0=phi0, bounds=((-6.28, 6.28), (-6.28, 6.28)))
            if phi_min.success:
                break
        if not phi_min.success:
            print('not converged')
        min_phi1 = phi_min.x[0]
        min_phi2 = phi_min.x[1]

        t1 = H1.dot([[np.cos(min_phi1)], [np.sin(min_phi1)], [1]])
        t2 = H2.dot([[np.cos(min_phi2)], [np.sin(min_phi2)], [1]])

        if plot:
            t = symbols('t')
            alpha = np.array([[cos(t)], [sin(t)], [1]])
            el_nu = sol1.H_perp.dot(alpha)
            el_met_nu = S.dot(el_nu)
            el_anu = sol2.H_perp.dot(alpha)
            el_met_anu = S.dot(el_anu)
            p = plot_parametric((el_nu[0][0], el_nu[1][0], (t, -5, 5)),
                                (el_anu[0][0], el_anu[1][0], (t, -5, 5)),
                                (el_met_nu[0][0], el_met_nu[1][0], (t, -5, 5)),
                                (el_met_anu[0][0], el_met_anu[1][0], (t, -5, 5)),
                                (t1[0][0] + (t2[0][0] - t1[0][0])*t, 
                                 t1[1][0] + (t2[1][0] - t1[1][0])*t,
                                 (t, 0, 1)),
                                legend=False,
                                show=False,
                                dvioptions=['-D','6000'])
            p.save(f'{fn[0]}_PDG_mass.{fn[1]}')
        try:
            dn = DoubleNu([b1_nusol.vec, b2_nusol.vec], [l1_nusol.vec, l2_nusol.vec], ev, mW*1000, mT*1000, mW*1000, mT*1000)
            print('not singular')
            if len(dn.nunu_s) != 0:
                distances = []
                for solutions in dn.nunu_s:
                    # print(solutions)
                    # print(solutions[0][0])
                    distances.append([find_dist(solutions[0], nu1), find_dist(solutions[1], nu2)])
                # print(distances)

                return True, min(distances, key=lambda x: x[0] + x[1])
        except Exception as e:
            print(e)
            pass
        # print('With minimal distance')
        try:
            result1 = sol1.H.dot(np.linalg.inv(sol1.H_perp)).dot(t1)
            # print('  nu1', nu1[0][0], nu1[1][0], nu1[2][0])
        except:
            result1 = None
            # print('  nu1 was not found')
        try:
            result2 = sol2.H.dot(np.linalg.inv(sol2.H_perp)).dot(S.dot(t1))
            # print('  nu2', nu2[0][0], nu2[1][0], nu2[2][0])
        except:
            result2 = None
            # print('  nu2 was not found')
        # print('real nus:')
        # print('  nu1', nu1.px, nu1.py, nu1.pz)
        # print('  nu2', nu2.px, nu2.py, nu2.pz)
        return False, [find_dist(result1, nu1), find_dist(result2, nu2)]

    def Strategy(self, event):
        leptons = [child for child in event.TopChildren if child.is_lep]
        self.event_type = f'{len(leptons)}lep'
        leptons = sorted(leptons, key=lambda x: x.pt, reverse=True)
        l1, l2 = leptons[:2]
        self.leps = [l1.pdgid, l2.pdgid]
        t1 = l1.Parent[0]
        t2 = l2.Parent[0]
        b1 = [child for child in t1.Children if child.is_b][0]
        b2 = [child for child in t2.Children if child.is_b][0]
        nu1 = [child for child in t1.Children if child.is_nu][0]
        nu2 = [child for child in t2.Children if child.is_nu][0]

        mT1 = (b1 + l1 + nu1).Mass
        mT2 = (b2 + l2 + nu2).Mass
        mW1 = (l1 + nu1).Mass
        mW2 = (l2 + nu2).Mass
        fake_event = Event()
        met0 = (nu1 + nu2).pt
        met_phi0 = (nu1 + nu2).phi
        fake_event.met = met0 
        fake_event.met_phi = met_phi0

        results = pyc.NuSol.Polar.NuNu([make_input(b1)], [make_input(b2)], [make_input(l1)], [make_input(l2)], [[met0, met_phi0]], [[mW*1000, mT*1000, 0]])
        if not results[-1][0] or len(results[0][0]) == 0:
            plot_dir_name = 'intersection_found'
        else:
            plot_dir_name = 'intersection_not_found'

        l1_nusol = ParticleNutest(l1.pt, l1.eta, l1.phi, l1.e)
        l2_nusol = ParticleNutest(l2.pt, l2.eta, l2.phi, l2.e)
        b1_nusol = ParticleNutest(b1.pt, b1.eta, b1.phi, b1.e)
        b2_nusol = ParticleNutest(b2.pt, b2.eta, b2.phi, b2.e)
        ev = EventNutest(met0, met_phi0)
        sol1 = NuSol(b1_nusol.vec, l1_nusol.vec, mT2=mT1**2, mW2=mW1**2)
        sol2 = NuSol(b2_nusol.vec, l2_nusol.vec, mT2=mT2**2, mW2=mW2**2)
        
        dn = DoubleNu([b1_nusol.vec, b2_nusol.vec], [l1_nusol.vec, l2_nusol.vec], ev, mW1, mT1, mW2, mT2)
        t = symbols('t')
        alpha = np.array([[cos(t)], [sin(t)], [1]])
        el_nu = dn.solutionSets[0].H_perp.dot(alpha)
        el_met_nu = dn.S.dot(el_nu)
        el_anu = dn.solutionSets[1].H_perp.dot(alpha)
        el_met_anu = dn.S.dot(el_anu)
        p = plot_parametric((el_nu[0][0], el_nu[1][0], (t, -5, 5)),
                            (el_anu[0][0], el_anu[1][0], (t, -5, 5)),
                            (el_met_nu[0][0], el_met_nu[1][0], (t, -5, 5)),
                            (el_met_anu[0][0], el_met_anu[1][0], (t, -5, 5)),
                            legend=False,
                            show=False,
                            dvioptions=['-D','6000'])
        p.save(f'Plots/{plot_dir_name}/{event.hash}_calculated_mass.png')
        sol1 = NuSol(b1_nusol.vec, l1_nusol.vec)
        sol2 = NuSol(b2_nusol.vec, l2_nusol.vec)
        V0 = np.outer([ev.vec.px, ev.vec.py, 0], [0, 0, 1])
        S = V0 - UnitCircle()
        H1 = sol1.H_perp
        H2 = S.dot(sol2.H_perp)
 
        def calculate_distance(phis):
            t1 = H1.dot([[np.cos(phis[0])], [np.sin(phis[0])], [1]])
            t2 = H2.dot([[np.cos(phis[1])], [np.sin(phis[1])], [1]])
            return ((t2[0][0] - t1[0][0])**2 + (t2[1][0] - t1[1][0])**2 + (t2[2][0] - t1[2][0])**2)**0.5
        
        phi0s = sum([[[p1, p2] for p1 in [0, 3.14/2, -3.14/2, 3.14, -3.14]] for p2 in [0, 3.14/2, -3.14/2, 3.14, -3.14]], start=[])

        for phi0 in phi0s:
            phi_min = scipy.optimize.minimize(calculate_distance, x0=phi0, bounds=((-6.28, 6.28), (-6.28, 6.28)))
            if phi_min.success:
                break

        min_phi1 = phi_min.x[0]
        min_phi2 = phi_min.x[1]

        t1 = H1.dot([[np.cos(min_phi1)], [np.sin(min_phi1)], [1]])
        t2 = H2.dot([[np.cos(min_phi2)], [np.sin(min_phi2)], [1]])

        alpha = np.array([[cos(t)], [sin(t)], [1]])
        el_nu = sol1.H_perp.dot(alpha)
        el_met_nu = S.dot(el_nu)
        el_anu = sol2.H_perp.dot(alpha)
        el_met_anu = S.dot(el_anu)
        p = plot_parametric((el_nu[0][0], el_nu[1][0], (t, -5, 5)),
                            (el_anu[0][0], el_anu[1][0], (t, -5, 5)),
                            (el_met_nu[0][0], el_met_nu[1][0], (t, -5, 5)),
                            (el_met_anu[0][0], el_met_anu[1][0], (t, -5, 5)),
                            (t1[0][0] + (t2[0][0] - t1[0][0])*t, 
                                t1[1][0] + (t2[1][0] - t1[1][0])*t,
                                (t, 0, 1)),
                            legend=False,
                            show=False,
                            dvioptions=['-D','6000'])
        p.save(f'Plots/{plot_dir_name}/{event.hash}_PDG_mass.png')

