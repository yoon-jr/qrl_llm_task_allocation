import torch
import pennylane as qml

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_policy(params, state):
    qml.RX(state[0]*0.1, wires=0)
    qml.RY(state[1]*0.1, wires=1)
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0,1])
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

class QRLAgent:
    def __init__(self):
        self.params = torch.randn(2, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.params], lr=0.1)
    def choose_action(self, state):
        return 0
    def learn(self, state, action, reward, next_state):
        self.optimizer.zero_grad()
        out = quantum_policy(self.params, torch.tensor(state[:2], dtype=torch.float32))
        loss = -reward * (out[0] + out[1] + 2)/4
        loss.backward()
        self.optimizer.step()