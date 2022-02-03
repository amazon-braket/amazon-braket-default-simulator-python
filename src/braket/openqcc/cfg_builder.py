class ControlFlowGraphNode:

    def __init__(self, subcircuit):
        self.subcircuit = subcircuit


class ControlFlowGraphBuilder:

    def __init__(self, qasm_string):
        self.cfg = self.build_cfg(qasm_string)

    def build_cfg(self, qasm_string):
        pass