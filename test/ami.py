from pyAMI.client import Client
from pyAMI.httpclient import HttpClient
import pyAMI_atlas.api as AtlasAPI
import http.client
import ssl

class HTTP(HttpClient):

    def __init__(self, config):
        super(HTTP, self).__init__(config)

    def connect(self, endpoint):
        self.endpoint = endpoint
        confx = {"host" : str(self.endpoint["host"]), "port" : int(self.endpoint["port"]), "context" : None}
        if self.endpoint['prot'] == 'http': pass
        elif self.endpoint['prot'] == 'https':
            confx["context"] = self.create_unverified_context()
            if self.config.conn_mode == self.config.CONN_MODE_LOGIN: pass
            else: confx["context"].load_cert_chain(certfile = self.config.cert_file, keyfile = self.config.key_file)
        self.connection = http.client.HTTPSConnection(**confx)

class ATLAS(Client):

    def __init__(self):
        super(ATLAS, self).__init__("atlas")
        self.httpClient = HTTP(self.config)


x = ATLAS()
name = "mc16_13TeV.412071.aMcAtNloPy8EG_A14_ttbar_hdamp258p75_dil_CFiltBVeto.deriv.DAOD_TOPQ1.e7129_a875_r10724_p4514"
print(AtlasAPI.get_dataset_info(x, name))

#client = pyAMI.client.Client("atlas")




#print(client.config.cert_file)
#AtlasAPI.list_datasets(client, patterns = ["mc12_8TeV.%"], fields = ["ldn"], type = "AOD")
