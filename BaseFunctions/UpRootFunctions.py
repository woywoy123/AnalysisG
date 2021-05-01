import uproot 


def FileObjectsToArrays(ObjectsFromFile_var):

    for key in ObjectsFromFile_var:
        for tree in ObjectsFromFile_var[key]:
            if isinstance(ObjectsFromFile_var[key][tree], list):
                Branch_Maps = ObjectsFromFile_var[key][tree][1]
                
                for b_key in Branch_Maps:
                    Branch_Maps[b_key] = Branch_Maps[b_key].array()

            if isinstance(ObjectsFromFile_var[key][tree], dict):
                Tree_Maps = ObjectsFromFile_var[key][tree]
                
                for t_key in Tree_Maps:
                    for b_key in Tree_Maps[t_key][1]:
                        ObjectsFromFile_var[key][tree][t_key][1][b_key] = ObjectsFromFile_var[key][tree][t_key][1][b_key].array()

