import torch
from muq import MuQ

device = 'cpu'

muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
muq = muq.to(device).eval()

class MuQWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        output = self.model(x, output_hidden_states=False)
        return output.last_hidden_state

wrapped_model = MuQWrapper(muq)

# --- STRICT STATIC SHAPE DEFINITION ---
# 24,000 Hz * 10 seconds = 240,000 samples
STATIC_SAMPLES = 240000
example_inputs = (torch.randn(1, STATIC_SAMPLES, device=device),)

print("Exporting model using Dynamo (Static 10s Shape)...")
onnx_program = torch.onnx.export(
    wrapped_model,
    args=example_inputs,
    dynamo=True,                  # Use torch.export backend logic
    dynamic_shapes=None,          # Removed to force a 100% static model graph
    input_names=['input_audio'],
    output_names=['last_hidden_state'],
    optimize=True,                # Leverages built-in graph optimization
    verify=False                  # Set to True if you want a runtime validation pass
)

# Save the returned ONNXProgram to disk
onnx_model_path = "muq_large_dynamo.onnx"
onnx_program.save(onnx_model_path)

print(f"Model successfully saved to {onnx_model_path}")
