import logging
from .code_similarity import *

class CodeSimManager:
    def __init__(self, code):
        self.code_sim_tool = SafeCodeSim(code)

    def get_similarity_score(self, code):
        return self.code_sim_tool.is_similar()
