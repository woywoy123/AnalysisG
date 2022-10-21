from .String import String 
from .General import Tools
class Tables(Tools):
    
    def __init__(self):
        self._S = " | "
        self._Columns = {}
        self._Rows = {}
        self._matrix = {}
        self.Sum = True
        self.Title = None

    def AddRowTitle(self, Title, Unit = False):
        self._Rows[Title] = Title
        if Unit:
            self._Rows[Title] += "(" + str(Unit) + ")"

    def AddColumnTitle(self, Title, Unit = False):
        self._Columns[Title] = Title
        if Unit:
            self._Columns[Title] += "(" + str(Unit) + ")"
    
    def AddValues(self, RowTitle, ColumnTitle, Val, RowUnit = False, ColUnit = False):
        RowTitle = str(RowTitle)
        ColumnTitle = str(ColumnTitle)
        self.AddRowTitle(RowTitle, RowUnit)
        self.AddColumnTitle(ColumnTitle, ColUnit)
        if RowTitle not in self._matrix:
            self._matrix[RowTitle] = {}
        if ColumnTitle not in self._matrix[RowTitle]:
            self._matrix[RowTitle][ColumnTitle] = []
        self._matrix[RowTitle][ColumnTitle].append(Val)
   
    def AlignColumns(self, inpt):
        cols = []
        for i in inpt:
            splitter = i.split(self._S)
            tmp = [len(k) for k in splitter]
            if len(cols) == 0:
                cols = tmp
            cols = [c if c >= s else s for s, c in zip(cols, tmp)]
        for i in range(len(inpt)):
            splitter = inpt[i].split(self._S)
            inpt[i] = self._S.join([s + " "*(k - len(s)) for s, k in zip(splitter, cols)])
        return inpt 


    def Compile(self):
        def Transformer(inpt, row, col):
            rinpt = ""
            cinpt = ""
            if row not in inpt:
                return ""
            if col not in inpt[row]:
                return ""

            out = sum(inpt[row][col]) if self.Sum else out
            return out
        
        def ColMargin():
            out_s = {}
            out_n = {}
            all_s = 0
            for row in self._matrix:
                s = sum(list(self.MergeNestedList(list(self._matrix[row].values()))))
                out_s[row] = s
                out_n[row] = s  
                all_s += s
            for row in out_n:
                out_n[row] = str(round(100*float(out_n[row] / all_s), 4))
            return out_s, out_n

        def RowMargin():
            o_s, _ = ColMargin()
            s = sum(list(self.MergeNestedList(list(o_s.values())))) 
            
            dic_col = {}
            for row in self._matrix:
                for col in self._matrix[row]:
                    if col not in dic_col:
                        dic_col[col] = 0
                    dic_col[col] += sum(self._matrix[row][col])
            dic_col_sum = {}
            for col in dic_col:
                dic_col_sum[col] = dic_col[col]
                dic_col[col] = round(100*float(dic_col[col] / s), 4)

            return dic_col_sum, dic_col

        rows = list(self._matrix)
        marg_row_s, marg_row_p = ColMargin()
        output = [self._S.join(self._Columns.keys()) + self._S + " Sum " + self._S + " fraction (%) "]
        for r in rows:
            s = r + self._S + self._S.join([str(Transformer(self._matrix, r, c)) for c in list(self._Columns)[1:]]) 
            s += self._S + str(marg_row_s[r]) + self._S + marg_row_p[r]
            output += [s]
        
        col_mrg_s, col_mrg_p = RowMargin()
        output += ["Sum of Column" + self._S + self._S.join([str(s) for s in col_mrg_s.values()]) + self._S + self._S]
        output += ["fraction (%)" + self._S + self._S.join([str(s) for s in col_mrg_p.values()]) + self._S + self._S]

        output = self.AlignColumns(output)

        spl = output[0].split(self._S)
        output.insert(1, "+".join(["-"*(len(s)+1 if i == 0 else len(s)+2) for i, s in zip(range(len(spl)), spl)]))
        output.insert(-2, "+".join(["-"*(len(s)+1 if i == 0 else len(s)+2) for i, s in zip(range(len(spl)), spl)]))
        output.insert(0, "="*(int(len(output[0])/2) - len(str(self.Title))) + " " + str(self.Title) + " " + "="*(int(len(output[0])/2)))
        self.output = output
    
    def DumpTableToFile(self, Directory):
        directory = "/".join(Directory.split("/")[:-1])
        self.mkdir(directory)
        Directory = self.AddTrailing(Directory, ".txt")
        F = open(Directory, "w")
        F.write("\n".join(self.output))
        F.close()
