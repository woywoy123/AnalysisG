from .Notification import Notification

class _Optimizer(Notification):
    def __init__(self): pass

    def _searchdatasplits(self, path):
        if path.endswith(".hdf5"): pass
        else: path += ".hdf5"

        msg = "No sample splitting found (k-Fold). Training on entire sample."
        if self.IsFile(path): pass
        else: return self.Warning(msg)

        train_fold = {}
        eval_fold = {}
        leaveout_fold = {}
        maps = self._cmod.length()
        for k in maps:
            if not k.startswith("train"): pass
            else: train_fold[int(k.split("-k")[-1])] = maps[k]

            if not k.startswith("eval"): pass
            else: eval_fold[int(k.split("-k")[-1])] = maps[k]

            if not k.startswith("leave-out"): pass
            else: leaveout_fold["-".join(k.split("-")[:-1])] = maps[k]

        self.Success("\n"+"="*25 + " k-Fold Statistics " + "="*25)
        key = "Found the following training k-Folds in > " + self.TrainingName + " < :" + "\n"
        for i in sorted(train_fold):
            key += ":: kFold - " + str(i) + " (" + str(train_fold[i]) + ")"
            if not i%5: key += "\n" 
        self.Success(key)

        key = "Found the following validation k-Folds in > " + self.TrainingName + " < :" + "\n"
        for i in sorted(eval_fold):
            key += ":: kFold - " + str(i) + " (" + str(eval_fold[i]) + ")"
            if not i%5: key += "\n" 
        self.Success(key)

        key = "Found leave-out sub-sample > " + self.TrainingName + " < -> "
        for i in sorted(leaveout_fold): key += "(" + str(leaveout_fold[i]) + ")"
        self.Success(key + "\n" + "="*25 + " End Statistics " + "="*25)

        if self.kFold is None: self._cmod.UseTheseFolds(list(train_fold))
        elif isinstance(self.kFold, int): self._cmod.UseTheseFolds([self.kFold])
        else: self._cmod.UseTheseFolds([i for i in self.kFold if isinstance(i, int)])

    def _findpriortraining(self):
        if self.Epoch is None: self.Epoch = 0

        path = self.WorkingPath + "/machine-learning/" + self.RunName
        if self.IsPath(path): pass
        else: self.mkdir(path)

        if not self.ContinueTraining: return
        max_map = {}
        msg = "No prior training was found under: " + path + ". Generating..."
        for f in self.lsFiles(path, ".pth"):
            ep, fold = f.lstrip(path).split("/")[:2]
            ep, fold = int(ep.split("-")[-1]), int(fold.split("-")[-1])
            if fold not in max_map: max_map[fold] = ep
            if max_map[fold] >= ep: continue
            max_map[fold] = ep

        if not len(max_map): return self.Warning(msg)
        for fold, ep in max_map.items():
            if fold not in self._kOps: continue
            if fold not in self._kModels: continue

            self._kOps[fold].KFold = fold
            self._kModels[fold].KFold = fold

            self._kOps[fold].Epoch = ep
            self._kModels[fold].Epoch = ep

            self._kOps[fold].load()
            self._kModels[fold].load()
        self.Epoch = max_map

    def _nographs(self): return self.Warning("No Sample Graphs found")

    def _nomodel(self):
        if self.Model is None: pass
        else: return False

        self.Warning("No Model was given.")
        return True

    def _notcompatible(self):
        self.Failure("Model not compatible with given input graph sample.")
        return False

    def _invalidoptimizer(self):
        self.Failure("Invalid Optimizer:" + self.Optimizer)
        return False

    def _invalidscheduler(self):
        self.Failure("Invalid Scheduler: " + self.Scheduler)
        return False

    def _showloss(self, epoch, kfold):
        if not self.DebugMode: return
        string = ["Epoch-kFold: " + str(epoch) + " - " + str(kfold)]

        if self._kOps[kfold].scheduler is None: lr = None
        else: lr = self._kOps[kfold].scheduler.get_last_lr()[0]
        if lr is None: pass
        else: string[-1] += " Current LR: {:.10f}".format(lr)
        self.Success("\n->".join(string))
        report = self._cmod.metric_plot.reportable()
        if not len(report["loss_train"]): return
        for i, j in report.items():
            try: len(j)
            except: continue
            if not len(j): continue
            print(i, j)





