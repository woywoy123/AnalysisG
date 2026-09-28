from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import pickle
import json

class task:

    def __init__(self):
        self.kfold = None
        self.epoch = None
        self.mode  = None
        self.model = None
        self.done  = None
        self.host  = None
    
    def __hash__(self):
        o  = str(self.kfold) + ":" + str(self.epoch)
        o += self.mode + self.model
        return hash(o)

class Database:

    def __init__(self):
        self.chkp = 0
        x = 0 
        self.tasks = open("sets.txt", "r").readlines()
        self.pending = {}
        for i in self.tasks:
            k = i.replace("\n", "")
            k = i.split(" ")
            tk = task()
            tk.model = k[1]
            tk.mode  = k[3]
            tk.epoch = int(k[5])
            tk.kfold = int(k[7])
            tk.done  = bool(k[9] == "True")
            self.tasks[x] = tk if not tk.done else None
            x += 1 
        self.tasks = [i for i in self.tasks if i is not None]
        self.training   = [i for i in self.tasks if i.mode == "training"]
        self.validation = [i for i in self.tasks if i.mode == "validation"]
        self.evaluation = [i for i in self.tasks if i.mode == "evaluation"]

db = Database()
try: db = pickle.load(open("db.pkl", "rb"))
except: print("DB initialize")

class TaskBroker(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        hst = parse_qs(urlparse(self.path).query).get("task_id", [None])[0]

        db.chkp+=1
        if   len(db.training):   tk = db.training.pop()
        elif len(db.validation): tk = db.validation.pop()
        else: tk = db.evaluation.pop()

        try: db.pending[hst]
        except: db.pending[hst] = tk.mode
        if db.pending[hst] == tk.mode:
            tsk = {
                    "kfold": tk.kfold, "epoch" : tk.epoch,
                    "model" : tk.model, "mode": tk.mode,
                    "status" : "run"
            }
            tk.host = hst
        else:
            db.training   += [tk] * tk.mode == "training"
            db.validation += [tk] * tk.mode == "validation"
            db.evaluation += [tk] * tk.mode == "evaluation"
            tsk = {"status" : "idl"}
        self.wfile.write(json.dumps(tsk).encode("utf-8"))

        print("_____ Pending: "   , len(db.pending)   , "____")
        print("_____ Training: "  , len(db.training)  , "____")
        print("_____ Validation: ", len(db.validation), "____")
        print("_____ Evaluation: ", len(db.evaluation), "____")
        if db.chkp % 20: return
        pickle.dump(db, open("db.pkl", "wb"))
        print("DB STASHED")

def run_server(port = 2301):
    httpd = HTTPServer(("0.0.0.0", port), TaskBroker)
    try: httpd.serve_forever()
    except KeyboardInterrupt: httpd.server_close()

if __name__ == "__main__":
    run_server()


