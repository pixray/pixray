import argparse


class LossInterface():
    def __init__(self, settings=None, device=None):
        self.device = device

    # a list of optional strings
    def instance_settings(self, arglist):
        pass
    
    @staticmethod
    def add_settings(parser):
        #add parser.add_argument() here
        return parser
    
    def help(self):
        parser = argparse.ArgumentParser()
        parser = self.add_settings(parser)
        helpstring = ""
        for d in parser._actions:
            helpstring = f"""parmeter name: {d.dest}\nHelp: {d.help}\nUse case: pixray.add_argument({d.dest}={d.default})"""
        return helpstring

    def parse_settings(self,args):
        #do stuff with args here
        return args
    
    def add_globals(self,args):
        lossglobals = {}
        return lossglobals

    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        loss = None
        return loss
    

