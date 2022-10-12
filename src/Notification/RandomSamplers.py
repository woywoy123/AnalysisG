from .Notification import Notification

class RandomSamplers(Notification):

    def __init__(self):
        pass

    def NotEnoughEvents(self, Given, Requested):
        self.Failure("More events are requested than available. Returning given events but shuffled.")
        self.Failure("Given: " + str(len(Given)) + " Requested: " + str(Requested))
