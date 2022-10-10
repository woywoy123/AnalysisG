from Tooling import Tools

class LogDumper(Tools):

    def __init__(self):
        self._S = " | "

    def AlignColumns(self, inpt):
        size = 0 
        for i in inpt:
            tmp = max([len(k) for k in i.split(self._S)])
            size = tmp if tmp > size else size
        output = []
        for i in inpt:
            new_str = i.split(self._S)
            output += [self._S.join([" "*(size - len(s)) + s for s in new_str])]
        return output, size

    def FixWhiteSpace(self, inpt):
        col = inpt[0].split(self._S)
        Cols = {i : "" for i in range(len(inpt))}
        for i in range(len(col)):
            _s = []
            for t in Cols:
                _s += [inpt[t].split(self._S)[i] + self._S]
            out, s = self.AlignColumns(_s)
            for t in Cols:
                Cols[t] += out[t].split(self._S)[0] + self._S
        return list(Cols.values())
        
    def __MakeTable(self, inpt, titles, MinMaxDict = None):

        segment = {}
        segment_H = {}
        xheading = list(inpt)[0]
        yheading = " | ".join([titles[i] + " (" + inpt[xheading][i] + ")" for i in range(len(inpt[xheading]))])
        output = [xheading + self._S + yheading]
        xData = list(inpt)[1:]
        for i in xData:
            output += [str(i)  + self._S + self._S.join([str(j) for j in inpt[i]])]
        output, size = self.AlignColumns(output)
        
        if MinMaxDict == None:
            return output
        
        Min = "Min" + "".join([self._S + str(MinMaxDict[x][1]) + " @ " +str(MinMaxDict[x][0]) for x in MinMaxDict])
        Max = "Max" + "".join([self._S + str(MinMaxDict[x][3]) + " @ " +str(MinMaxDict[x][2]) for x in MinMaxDict])
        output.insert(1, Min)
        output.insert(2, Max)
        output, size = self.AlignColumns(output)
        output.insert(1, "_"*len(output[0]))
        output.insert(4, "-"*len(output[0])) 

        return output

    def DumpTLine(self, fig):
        x = fig.xData
        y = fig.yData
        if fig.DoStatistics:
            x = list(x)
        dic = {}
        dic[fig.xTitle] = fig.yTitle
        for i, j in zip(x, y):
            dic[i] = j

        MinMaxDict = {}
        Min = min(y)
        Max = max(y)
        IndexMin = x[y.index(Min)]
        IndexMax = x[y.index(Max)]
        MinMaxDict[fig.Title] = [IndexMin, Min, IndexMax, Max]
        return dic, MinMaxDict

    def MergeDicts(self, dics, MinMaxDict = None):
        titles = list(dics) 
        Cols = {}
        for t in titles:
            keys = list(dics[t])
            for k in keys:
                if k not in Cols:
                    Cols[k] = []
                Cols[k] += [dics[t][k]]
        return self.__MakeTable(Cols, titles, MinMaxDict)

    def DumpSummaryTable(self, Dict):

        Cols = list(Dict[list(Dict)[0]])
        SumScore = {M : 0 for M in Cols}
        Rows = [" "] + list(Dict)
        Total = [] 
        for M in Cols:
            Dict[" "] = {M :  M + self._S + "Min" + self._S + "Max" + self._S + "Score" + self._S}
            Output = []
            for row in Rows:
                Line = row + self._S
                val = Dict[row][M]
                if row == " ":
                    Line += val
                else:
                    Line += "Epoch (Min, Max): " + str(val["MinMax"][0]) + ", " + str(val["MinMax"][2]) + self._S 
                    Line += str(val["MinMax"][1]) + self._S + str(val["MinMax"][3])
                    Line += str(self._S + str(val["Score"]) + self._S)
                    SumScore[M] += val["Score"]
                Output += [Line]
            
            Output = self.FixWhiteSpace(Output)
            Output += ["="*len(Output[0])] 
            Output.insert(1, "="*len(Output[0]))
            
            Total += Output

        Out = [self._S + "Scores"]
        Out += [i + self._S + str(SumScore[i]) for i in SumScore]
        Total += self.FixWhiteSpace(Out)
        return Total

    def DumpTLines(self, figs):
        lines = {}
        MinMax = {}
        for i in range(len(figs)):
            lines[figs[i].Title], MinMaxDict = self.DumpTLine(figs[i])
            MinMax |= MinMaxDict
        return self.MergeDicts(lines, MinMax)


