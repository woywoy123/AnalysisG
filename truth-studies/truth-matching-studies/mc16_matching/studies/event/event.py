from AnalysisG.Templates import SelectionTemplate

class TruthEvent(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)

        self.met_data = {
                "delta_met_nus" : {}, "delta_met_children" : {},
                "truth-nus" : {}, "num_neutrino" : {}, "met" : {}
        }

        self.num_leptons = {
                "0L" : 0, "1L" : 0, "2L" : 0, "3L" : 0, "4L" : 0, "2LOS" : 0, "2LSS" : 0
        }

        self.num_tops = []

        self.met_cartesian = {
            "met_x" : {"0L" : [], "1L" : [], "2L" : [], "3L" : [], "4L" : [], "2LOS" : [], "2LSS" : []},
            "met_y" : {"0L" : [], "1L" : [], "2L" : [], "3L" : [], "4L" : [], "2LOS" : [], "2LSS" : []},
            "delta_met_x" : {"0L" : [], "1L" : [], "2L" : [], "3L" : [], "4L" : [], "2LOS" : [], "2LSS" : []},
            "delta_met_y" : {"0L" : [], "1L" : [], "2L" : [], "3L" : [], "4L" : [], "2LOS" : [], "2LSS" : []}
        }

    def Selection(self, event): return True

    def Strategy(self, event):
        self.num_tops += [len(event.Tops)]

        met, phi = event.met, event.met_phi

        nus = [i for i in event.TopChildren if i.is_nu]
        all_children = sum(event.TopChildren)

        n_nus = len(nus)
        try: self.met_data["met"][n_nus] += [met/1000]
        except KeyError: self.met_data["met"][n_nus] = [met/1000]

        met_nu = sum(nus).pt if n_nus else 0
        met_nu = abs(met_nu - met)/1000
        try: self.met_data["delta_met_nus"][n_nus] += [met_nu]
        except KeyError: self.met_data["delta_met_nus"][n_nus] = [met_nu]

        delta_met = abs(all_children.pt - met)/1000
        try: self.met_data["delta_met_children"][n_nus] += [delta_met]
        except KeyError: self.met_data["delta_met_children"][n_nus] = [delta_met]

        try: self.met_data["num_neutrino"][n_nus] += [n_nus]
        except KeyError: self.met_data["num_neutrino"][n_nus] = [n_nus]

        nu_met = sum(nus).pt/1000 if n_nus else 0
        try: self.met_data["truth-nus"][n_nus] += [nu_met]
        except KeyError: self.met_data["truth-nus"][n_nus] = [nu_met]

        leps = [c for t in event.Tops for c in t.Children if c.is_lep and not c.is_nu]
        header = str(len(leps)) + "L"
        self.num_leptons[header] += 1

        met_x, met_y = self.Px(met, phi), self.Py(met, phi)
        all_x, all_y = all_children.px, all_children.py

        self.met_cartesian["met_x"][header] += [met_x]
        self.met_cartesian["met_y"][header] += [met_y]
        self.met_cartesian["delta_met_x"][header] += [abs(all_x - met_x)/1000]
        self.met_cartesian["delta_met_y"][header] += [abs(all_y - met_y)/1000]

        if len(leps) == 2:
            mode = ""
            if leps[0].charge == leps[1].charge: mode = "SS"
            else: mode = "OS"
            self.num_leptons[header + mode] += 1

            self.met_cartesian["met_x"][header + mode] += [met_x]
            self.met_cartesian["met_y"][header + mode] += [met_y]
            self.met_cartesian["delta_met_x"][header + mode] += [abs(all_x - met_x)/1000]
            self.met_cartesian["delta_met_y"][header + mode] += [abs(all_y - met_y)/1000]
