import urllib.request
import socket
import json

def fetch_task(task_id):
    url = f"http://localhost:2301/?task_id={task_id}"
    rep = urllib.request.urlopen(url)
    data = json.loads(rep.read().decode("utf-8"))
    print(data)

if __name__ == "__main__":
    hostname = socket.gethostname()
    fetch_task(hostname)
