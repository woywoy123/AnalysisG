#include "mc20_fuzzy.h"

mc20_fuzzy::mc20_fuzzy(){this -> name = "mc20_fuzzy";}
mc20_fuzzy::~mc20_fuzzy(){}

selection_template* mc20_fuzzy::clone(){
    return (selection_template*)new mc20_fuzzy();
}

void mc20_fuzzy::merge(selection_template* sl){
    mc20_fuzzy* slt = (mc20_fuzzy*)sl; 
    merge_data(&this -> output, &slt -> output); 
}

bool mc20_fuzzy::selection(event_template* ev){return true;}

bool mc20_fuzzy::strategy(event_template* ev){
    exp_mc20* evn = (exp_mc20*)ev; 
    std::vector<particle_template*> dtops = evn -> Tops; 
    std::vector<particle_template*> physT = evn -> PhysicsTruth; 
    std::vector<particle_template*> jets  = evn -> Jets; 
    std::vector<particle_template*> physD = evn -> PhysicsDetector; 

    this -> output.push_back(packet_t()); 
    packet_t* pkl = &this -> output[0]; 

    for (size_t x(0); x < dtops.size(); ++x){
        pkl -> truth_tops.push_back(new particle(dtops[x], true));
        std::vector<particle_template*> ch_ = this -> vectorize(&dtops[x] -> children); 
        if (!ch_.size()){continue;}

        std::vector<particle_template*> nu = {}; 
        std::vector<particle_template*> lep = {};
        for (size_t y(0); y < ch_.size(); ++y){
            if (ch_[y] -> is_nu){nu.push_back(ch_[y]);}
            if (ch_[y] -> is_lep){lep.push_back(ch_[y]);}
        }

        particle_template* ptr = nullptr; 
        this -> sum(&ch_, &ptr);
        pkl -> children_tops.push_back(new particle(ptr, true)); 

        std::vector<particle_template*> tjets = {}; 
        for (size_t y(0); y < physT.size(); ++y){
            std::map<std::string, particle_template*> prnt = physT[y] -> parents; 
            if (!prnt.count(dtops[x] -> hash)){continue;}
            tjets.push_back(physT[y]); 
        }

        if (tjets.size()){
            merge_data(&tjets, &nu); 
            ptr = nullptr; 
            this -> sum(&tjets, &ptr);
            pkl -> truth_jets.push_back(new particle(ptr, true)); 
        }


        std::vector<particle_template*> _jets = {}; 
        for (size_t y(0); y < jets.size(); ++y){
            std::map<std::string, particle_template*> prnt = jets[y] -> parents; 
            if (!prnt.count(dtops[x] -> hash)){continue;}
            _jets.push_back(jets[y]); 
        }

        if (_jets.size()){
            ptr = nullptr; 
            merge_data(&_jets, &nu); 
            merge_data(&_jets, &lep); 
            this -> sum(&_jets, &ptr);
            pkl -> jets_children.push_back(new particle(ptr, true)); 
        }

        _jets = {}; 
        for (size_t y(0); y < physD.size(); ++y){
            std::map<std::string, particle_template*> prnt = physD[y] -> parents; 
            if (!prnt.count(dtops[x] -> hash)){continue;}
            _jets.push_back(physD[y]); 
        }

        if (_jets.size()){
            ptr = nullptr; 
            merge_data(&_jets, &nu); 
            this -> sum(&_jets, &ptr);
            pkl -> jets_leptons.push_back(new particle(ptr, true)); 
        }
    }
//        tops = event.Tops
//        self.truth_top += [t.Mass/1000 for t in tops]
//
//        for t in tops:
//            ch = t.Children
//            if not len(ch): continue
//
//            ch_ = [c for c in ch if c.is_lep]
//            self.truth_children["all"] += [sum(ch).Mass/1000]
//
//            # Children
//            mode = "had"
//            if len(ch_): mode = "lep"
//            self.truth_children[mode] += [sum(ch).Mass/1000]
//
//            # Truth Jets with Truth Children (leptonic decay)
//            tru_dic = {t : {}}
//            for tj in event.PhysicsTruth:
//                for ti in tj.Parent:
//                    try: tru_dic[ti][tj] = None
//                    except KeyError: continue
//
//            phys_tru = list(set(list(tru_dic[t])))
//            ntj = str(len(phys_tru)) + " - Truth Jets"
//            phys_tru = phys_tru + [c for c in ch if c.is_nu]
//            if len(phys_tru):
//                top_M = sum(phys_tru).Mass/1000
//                self.truth_jets["all"] += [top_M]
//                self.truth_jets[mode] += [top_M]
//                if len(ch_):
//                    if ntj not in self.n_truth_jets_lep: self.n_truth_jets_lep[ntj] = []
//                    self.n_truth_jets_lep[ntj] += [top_M]
//                else:
//                    if ntj not in self.n_truth_jets_had: self.n_truth_jets_had[ntj] = []
//                    self.n_truth_jets_had[ntj] += [top_M]
//
//            # Jets with Truth Children (leptonic decay)
//            jet_dic = {t : []}
//            for jet in event.Jets:
//                if t not in jet.Parent: continue
//                jet_dic[t] += [jet]
//
//            phys_det = list(set(list(jet_dic[t])))
//            nj = str(len(phys_det)) + " - Jets"
//            phys_det = phys_det + [c for c in ch if c.is_lep or c.is_nu]
//            if len(phys_det):
//                top_M = sum(phys_det).Mass/1000
//                self.jets_truth_leps["all"] += [top_M]
//                self.jets_truth_leps[mode] += [top_M]
//                if len(ch_):
//                    if nj not in self.n_jets_lep: self.n_jets_lep[nj] = []
//                    self.n_jets_lep[nj] += [top_M]
//                else:
//                    if nj not in self.n_jets_had: self.n_jets_had[nj] = []
//                    self.n_jets_had[nj] += [top_M]
//
//
//
//            # Detector only objects (except neutrinos)
//            jet_dic = {t : {}}
//            for jet in event.PhysicsDetector:
//                for ti in jet.Parent:
//                    try: jet_dic[ti][jet] = None
//                    except KeyError: continue
//
//            phys_det = list(jet_dic[t]) + [c for c in ch if c.is_nu]
//            phys_det = list(set(phys_det))
//            if len(phys_det):
//                top_M = sum(phys_det).Mass/1000
//                self.jets_leps["all"] += [top_M]
//                self.jets_leps[mode] += [top_M]
//
//










    return true; 
}

