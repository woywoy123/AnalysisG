from BaseFunctions.IO import *
import threading 

def ReadLeafsFromResonance(file_dir):

    def FindResonancePoints(truth_top_pt_obj, top_FromRes_obj, File):
        print("Starting: " +  File)
        if (top_FromRes_obj.num_entries == truth_top_pt_obj.num_entries):
            for i, j in zip(top_FromRes_obj.iterate(step_size = 1), truth_top_pt_obj.iterate(step_size = 1)):
                list_Res = i[0]["top_FromRes"]
                list_topPt = j[0]["truth_top_pt"]
                if list(set(list_Res)) != [0]:
                    print(list_Res, list_topPt)
            
            print("Searched: " + File)


    Files = ListSampleDirectories(file_dir)
    Entry_Objects = ObjectsFromFile(Files, "nominal", ["top_FromRes", "truth_top_pt"])
   
    threads = []
    for key in Entry_Objects:
        
        for f in Entry_Objects[key]:
            tree_list = Entry_Objects[key][f]
            tree_obj = tree_list["nominal"][0]
            branch_map = tree_list["nominal"][1]

            top_FromRes_obj = branch_map["top_FromRes"]
            truth_top_pt_obj = branch_map["truth_top_pt"]
           
            x = threading.Thread(target = FindResonancePoints, args=(truth_top_pt_obj, top_FromRes_obj, f))    
            threads.append(x)
            x.start()
            
            if len(threads) == 4:
                for i in range(len(threads)):
                    threads[i].join()
                    threads.pop(i)

                    if len(threads) != 4:
                        break
        











            #if (top_FromRes_obj.num_entries == truth_top_pt_obj.num_entries):
            #    for i, j in zip(top_FromRes_obj.iterate(step_size = 1), truth_top_pt_obj.iterate(step_size = 1)):
            #        list_Res = i[0]["top_FromRes"]
            #        list_topPt = j[0]["truth_top_pt"]
            #        if list(set(list_Res)) != [0]:
            #            print(list_Res, list_topPt)
            #        


        








