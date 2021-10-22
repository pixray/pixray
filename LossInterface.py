import argparse
from torch import nn


class LossInterface(nn.Module):
    def __init__(self):
        super().__init__()
    
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
        globals = {}
        return globals

    def forward(self, z):
        loss = None
        return loss
    

