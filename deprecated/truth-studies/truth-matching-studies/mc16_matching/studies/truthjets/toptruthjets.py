from AnalysisG.Templates import SelectionTemplate

class TopTruthJets(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)

        self.top_mass = {
                "hadronic" : [], "leptonic" : [],
                "ntruthjets" : {"hadronic" : {}, "leptonic" : {}},
                "merged_tops" : {"hadronic" : {}, "leptonic" : {}}
        }

        self.ntops_lost = []

        self.truthjet_top = {
                "resonant-leptonic"  : {"dr" : [], "energy" : [], "pt" : []},
                "spectator-leptonic" : {"dr" : [], "energy" : [], "pt" : []},
                "resonant-hadronic"  : {"dr" : [], "energy" : [], "pt" : []},
                "spectator-hadronic" : {"dr" : [], "energy" : [], "pt" : []},
                "background" : {"dr" : []}
        }

        self.truthjet_partons = {
                "resonant-leptonic"  : {},
                "spectator-leptonic" : {},
                "resonant-hadronic"  : {},
                "spectator-hadronic" : {},
                "background" : {}
        }

        self.truthjets_contribute = {
                "all" : {"energy" : [], "pt" : [], "n-partons" : []},
                "n-tops" : {}
        }

        self.truthjet_mass = {"all" : [], "n-tops" : []}

    def Selection(self, event): return True

    def Strategy(self, event):

        ######### basic truth top matching to jets ############
        lost = 0
        for t in event.Tops:
            frac = []
            frac += t.TruthJets
            if not len(frac): lost += 1; continue
            frac += [c for c in t.Children if c.is_lep or c.is_nu]
            top_mass = sum(frac).Mass/1000

            n_truj = len(t.TruthJets)
            n_tops = len(set([t_ for tj in t.TruthJets for t_ in tj.Tops]))

            is_lep = len([c for c in t.Children if c.is_lep]) != 0
            mode = "leptonic" if is_lep else "hadronic"
            self.top_mass[mode] += [top_mass]
            if n_truj not in self.top_mass["ntruthjets"][mode]:
                self.top_mass["ntruthjets"][mode][n_truj] = []
            self.top_mass["ntruthjets"][mode][n_truj] += [top_mass]

            if n_tops not in self.top_mass["merged_tops"][mode]:
                self.top_mass["merged_tops"][mode][n_tops] = []
            self.top_mass["merged_tops"][mode][n_tops] += [top_mass]
        self.ntops_lost += [lost]

        ########## kinematic studies ##########
        for t in event.Tops:
            checked = []
            mode = "resonant" if t.FromRes else "spectator"
            decay = "leptonic" if len([c for c in t.Children if c.is_nu or c.is_lep]) else "hadronic"
            mode = mode + "-" + decay
            for tj1 in t.TruthJets:
                for tj2 in t.TruthJets:
                    if tj1 == tj2: continue
                    if tj2 in checked: continue

                    self.truthjet_top[mode]["dr"].append(tj1.DeltaR(tj2))
                    self.truthjet_top[mode]["energy"].append(t.e/1000)
                    self.truthjet_top[mode]["pt"].append(t.pt/1000)
                checked.append(tj1)

                for tj2 in event.TruthJets:
                    if tj2 in t.TruthJets: continue
                    self.truthjet_top["background"]["dr"].append(tj2.DeltaR(tj1))

                for prt in tj1.Parton:
                    symbol = "null" if not len(prt.symbol) else prt.symbol
                    if symbol not in self.truthjet_partons[mode]:
                        self.truthjet_partons[mode][symbol] = {}
                        self.truthjet_partons[mode][symbol]["dr"] = []
                        self.truthjet_partons[mode][symbol]["top-pt"] = []
                        self.truthjet_partons[mode][symbol]["top-energy"] = []
                        self.truthjet_partons[mode][symbol]["parton-pt"] = []
                        self.truthjet_partons[mode][symbol]["parton-energy"] = []
                        self.truthjet_partons[mode][symbol]["truthjet-pt"] = []
                        self.truthjet_partons[mode][symbol]["truthjet-energy"] = []

                    dr = prt.DeltaR(tj1)
                    self.truthjet_partons[mode][symbol]["dr"].append(dr)
                    self.truthjet_partons[mode][symbol]["top-pt"].append(t.pt/1000)
                    self.truthjet_partons[mode][symbol]["top-energy"].append(t.e/1000)

                    self.truthjet_partons[mode][symbol]["parton-pt"].append(prt.pt/1000)
                    self.truthjet_partons[mode][symbol]["parton-energy"].append(prt.e/1000)

                    self.truthjet_partons[mode][symbol]["truthjet-pt"].append(tj1.pt/1000)
                    self.truthjet_partons[mode][symbol]["truthjet-energy"].append(tj1.e/1000)





        mode = "background"
        for tj in event.TruthJets:
            if len(tj.Tops): continue
            for prt in tj.Parton:
                symbol = "null" if not len(prt.symbol) else prt.symbol
                if symbol not in self.truthjet_partons[mode]:
                    self.truthjet_partons[mode][symbol] = {}
                    self.truthjet_partons[mode][symbol]["dr"] = []
                    self.truthjet_partons[mode][symbol]["parton-pt"] = []
                    self.truthjet_partons[mode][symbol]["parton-energy"] = []
                    self.truthjet_partons[mode][symbol]["truthjet-pt"] = []
                    self.truthjet_partons[mode][symbol]["truthjet-energy"] = []

                dr = prt.DeltaR(tj)
                self.truthjet_partons[mode][symbol]["dr"].append(dr)

                self.truthjet_partons[mode][symbol]["parton-pt"].append(prt.pt/1000)
                self.truthjet_partons[mode][symbol]["parton-energy"].append(prt.e/1000)

                self.truthjet_partons[mode][symbol]["truthjet-pt"].append(tj.pt/1000)
                self.truthjet_partons[mode][symbol]["truthjet-energy"].append(tj.e/1000)


        ####### Ghost Parton Energy constributions ############
        for tj in event.TruthJets:
            pt_sum = sum([prt.pt for prt in tj.Parton])
            if not pt_sum: continue
            self.truthjets_contribute["all"]["pt"] += [tj.pt/pt_sum]
            e_sum = sum([prt.e for prt in tj.Parton])
            self.truthjets_contribute["all"]["energy"] += [tj.e/e_sum]
            self.truthjets_contribute["all"]["n-partons"] += [len(tj.Parton)]
            self.truthjet_mass["all"] += [tj.Mass/1000]
            self.truthjet_mass["n-tops"] += [len(tj.Tops)]

            # This part checks the energy contribution of ghost partons matched to top children
            # and what happens to the reconstructed top-mass when some low energy contributions are 
            # removed.
            alls = []
            top_maps = {}
            for prt in tj.Parton:
                for tc in prt.Parent:
                    top_ = tc.Parent[0]
                    if top_ not in top_maps: top_maps[top_] = []
                    top_maps[top_] += [prt]
                    alls += [prt]
                    break

            n_top = len(top_maps)
            if not n_top: continue
            if n_top not in self.truthjets_contribute["n-tops"]:
                self.truthjets_contribute["n-tops"][n_top] = {"energy_r" : []}

            if not len(alls): continue
            total = sum(set(alls)).e
            for tx in top_maps:
                r = sum(set(top_maps[tx])).e/total
                self.truthjets_contribute["n-tops"][n_top]["energy_r"] += [r]

