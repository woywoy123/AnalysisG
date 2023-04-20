from AnalysisG.Templates import SelectionTemplate

class Example(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)
        self.Top = {"Truth" : []}
        self.Children = {"Truth" : []}

    def Strategy(self, event):
        self.Top["Truth"] += [t.Mass for t in event.Tops]
        self.Children["Truth"] += [c.Mass for c in event.TopChildren]
        
        return "Success->Example"

class Example2(SelectionTemplate):

    def __init__(self):
        SelectionTemplate.__init__(self)
        self.Top = {"Truth" : []}
        self.Children = {"Truth" : []}

    def Strategy(self, event):
        self.Top["Truth"] += [t.Mass for t in event.Tops]
        self.Children["Truth"] += [c.Mass for c in event.TopChildren]
        self.Children["Test"] = {}
        self.Children["Test"]["t"] = [1]
        
        return "Success->Example"

