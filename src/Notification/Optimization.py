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
        for i in self._cmod.kFolds: print(i)

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


#    def _searchtraining(self):
#        self.Epoch = 0
#        pth = self._outDir + "/" + self.RunName
#        epochs = self.ls(pth)
#        kf = []
#        if not self.ContinueTraining:
#            return True
#        if len(epochs) == 0:
#            return self.Warning(
#                "No prior training was found under: " + pth + ". Generating..."
#            )
#        epochs = [int(ep) for ep in epochs]
#        epochs.sort()
#        for ep in epochs:
#            path = pth + "/" + str(ep) + "/"
#            for k in self.ls(path):
#                kPath = path + k + "/TorchSave.pth"
#                if not self.IsFile(kPath):
#                    continue
#                if k not in self._kModels:
#                    continue
#                self._kModels[k]._pth = self._outDir + "/" + self.RunName
#                self._kOp[k]._pth = self._outDir + "/" + self.RunName
#                self._kModels[k].Epoch = str(ep) + "/" + k
#                self._kOp[k].Epoch = str(ep) + "/" + k
#
#        for i in self._kModels:
#            if self._kModels[i].Epoch is None:
#                self._kModels[i].Epoch = ""
#                continue
#            try:
#                self.Success("Model loaded: " + self._kModels[i].load())
#            except KeyError:
#                return not self.Warning("Loading Model Failed. Skipping loading...")
#            try:
#                self.Success("Optimizer loaded: " + self._kOp[i].load())
#            except KeyError:
#                return not self.Warning("Loading Optimizer Failed. Skipping loading...")
#        return True
#
#    def _showloss(self):
#        string = ["Epoch-kFold: " + self._ep.Epoch]
#        if self._op._sc is not None:
#            string[-1] += " Current LR: {:.10f}".format(self._op._sc.get_lr()[0])
#        for f in self.Model._l:
#            string.append(
#                "Feature: {}, Loss: {:.10f}, Accuracy: {:.10f}".format(
#                    f, self.Model._l[f]["loss"], self.Model._l[f]["acc"]
#                )
#            )
#        print("\n-> ".join(string))
