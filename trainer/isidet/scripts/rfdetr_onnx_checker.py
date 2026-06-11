import onnx
import onnxsim
import onnxruntime as ort

MODEL_PATH = "isidet/models/rfdetr/29-03-2026_0154/inference_model.onnx"

print("\n===== 🔧 ENVIRONMENT CHECK =====")
print("onnx version       :", onnx.__version__)
print("onnxsim version    :", onnxsim.__version__)
print("onnxruntime version:", ort.__version__)

print("\n===== 📦 MODEL CHECK =====")
model = onnx.load(MODEL_PATH)

print("IR version         :", model.ir_version)
print("Producer           :", model.producer_name)

print("\nInputs:")
for i in model.graph.input:
    print(" -", i.name)

print("\nOutputs:")
for o in model.graph.output:
    print(" -", o.name)

print("\n===== ✅ ONNX CHECKER =====")
try:
    onnx.checker.check_model(model)
    print("✔ ONNX model is VALID")
except Exception as e:
    print("❌ ONNX checker failed:", e)

print("\n===== ⚡ ONNXSIM TEST =====")
try:
    model_simp, check = onnxsim.simplify(model)
    print("✔ Simplification success:", check)
except Exception as e:
    print("❌ Simplification failed:", e)
