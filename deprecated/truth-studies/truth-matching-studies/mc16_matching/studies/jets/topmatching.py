from AnalysisG.Templates import SelectionTemplate

class TopJets(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)

        self.top_mass = {
                "hadronic"    : [], "leptonic" : [],
                "njets"       : {"hadronic" : {}, "leptonic" : {}},
                "merged_tops" : {"hadronic" : {}, "leptonic" : {}}
        }

        self.ntops_lost = []

        self.jet_top = {
                "resonant-leptonic"  : {"dr" : [], "energy" : [], "pt" : []},
                "spectator-leptonic" : {"dr" : [], "energy" : [], "pt" : []},
                "resonant-hadronic"  : {"dr" : [], "energy" : [], "pt" : []},
                "spectator-hadronic" : {"dr" : [], "energy" : [], "pt" : []},
                "background" : {"dr" : []}
        }

        self.jet_partons = {
                "resonant-leptonic"  : {},
                "spectator-leptonic" : {},
                "resonant-hadronic"  : {},
                "spectator-hadronic" : {},
                "background" : {}
        }

        self.jets_contribute = {
                "all" : {"energy" : [], "pt" : [], "n-partons" : []},
                "n-tops" : {}
        }

        self.jet_mass = {"all" : [], "n-tops" : []}

    def Strategy(self, event):

        ######### basic truth top matching to jets ############
        lost = 0
        for t in event.Tops:
            frac = []
            frac += t.Jets
            if not len(frac): lost += 1; continue
            frac += [c for c in t.Children if c.is_lep or c.is_nu]
            top_mass = sum(frac).Mass/1000

            n_truj = len(t.Jets)
            n_tops = len(set([t_ for j in t.Jets for t_ in j.Tops]))

            is_lep = len([c for c in t.Children if c.is_lep]) != 0
            mode = "leptonic" if is_lep else "hadronic"
            self.top_mass[mode] += [top_mass]
            if n_truj not in self.top_mass["njets"][mode]:
                self.top_mass["njets"][mode][n_truj] = []
            self.top_mass["njets"][mode][n_truj] += [top_mass]

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
            for j1 in t.Jets:
                for j2 in t.Jets:
                    if j1 == j2: continue
                    if j2 in checked: continue

                    self.jet_top[mode]["dr"].append(j1.DeltaR(j2))
                    self.jet_top[mode]["energy"].append(t.e/1000)
                    self.jet_top[mode]["pt"].append(t.pt/1000)
                checked.append(j1)

                for j2 in event.Jets:
                    if j2 in t.Jets: continue
                    self.jet_top["background"]["dr"].append(j2.DeltaR(j1))

                for prt in j1.Parton:
                    symbol = "null" if not len(prt.symbol) else prt.symbol
                    if symbol not in self.jet_partons[mode]:
                        self.jet_partons[mode][symbol] = {}
                        self.jet_partons[mode][symbol]["dr"] = []
                        self.jet_partons[mode][symbol]["top-pt"] = []
                        self.jet_partons[mode][symbol]["top-energy"] = []
                        self.jet_partons[mode][symbol]["parton-pt"] = []
                        self.jet_partons[mode][symbol]["parton-energy"] = []
                        self.jet_partons[mode][symbol]["jet-pt"] = []
                        self.jet_partons[mode][symbol]["jet-energy"] = []

                    dr = prt.DeltaR(j1)
                    self.jet_partons[mode][symbol]["dr"].append(dr)
                    self.jet_partons[mode][symbol]["top-pt"].append(t.pt/1000)
                    self.jet_partons[mode][symbol]["top-energy"].append(t.e/1000)

                    self.jet_partons[mode][symbol]["parton-pt"].append(prt.pt/1000)
                    self.jet_partons[mode][symbol]["parton-energy"].append(prt.e/1000)

                    self.jet_partons[mode][symbol]["jet-pt"].append(j1.pt/1000)
                    self.jet_partons[mode][symbol]["jet-energy"].append(j1.e/1000)

        mode = "background"
        for j in event.Jets:
            if len(j.Tops): continue
            for prt in j.Parton:
                symbol = "null" if not len(prt.symbol) else prt.symbol
                if symbol not in self.jet_partons[mode]:
                    self.jet_partons[mode][symbol] = {}
                    self.jet_partons[mode][symbol]["dr"] = []
                    self.jet_partons[mode][symbol]["parton-pt"] = []
                    self.jet_partons[mode][symbol]["parton-energy"] = []
                    self.jet_partons[mode][symbol]["jet-pt"] = []
                    self.jet_partons[mode][symbol]["jet-energy"] = []

                dr = prt.DeltaR(j)
                self.jet_partons[mode][symbol]["dr"].append(dr)

                self.jet_partons[mode][symbol]["parton-pt"].append(prt.pt/1000)
                self.jet_partons[mode][symbol]["parton-energy"].append(prt.e/1000)

                self.jet_partons[mode][symbol]["jet-pt"].append(j.pt/1000)
                self.jet_partons[mode][symbol]["jet-energy"].append(j.e/1000)


        ####### Ghost Parton Energy constributions ############
        for j in event.Jets:
            pt_sum = sum([prt.pt for prt in j.Parton])
            if not pt_sum: continue
            self.jets_contribute["all"]["pt"] += [j.pt/pt_sum]
            e_sum = sum([prt.e for prt in j.Parton])
            self.jets_contribute["all"]["energy"] += [j.e/e_sum]
            self.jets_contribute["all"]["n-partons"] += [len(j.Parton)]
            self.jet_mass["all"] += [j.Mass/1000]
            self.jet_mass["n-tops"] += [len(j.Tops)]

            # This part checks the energy contribution of ghost partons matched to top children
            # and what happens to the reconstructed top-mass when some low energy contributions are 
            # removed.
            alls = []
            top_maps = {}
            for prt in j.Parton:
                for tc in prt.Parent:
                    top_ = tc.Parent[0]
                    if top_ not in top_maps: top_maps[top_] = []
                    top_maps[top_] += [prt]
                    alls += [prt]
                    break

            n_top = len(top_maps)
            if not n_top: continue
            if n_top not in self.jets_contribute["n-tops"]:
                self.jets_contribute["n-tops"][n_top] = {"energy_r" : []}

            if not len(alls): continue
            total = sum(set(alls)).e
            for tx in top_maps:
                r = sum(set(top_maps[tx])).e/total
                self.jets_contribute["n-tops"][n_top]["energy_r"] += [r]


