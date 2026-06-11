import onnxruntime as ort

def check_model(path):
    print(f"\n🔍 Interrogating: {path}")
    try:
        session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        
        print("--- INPUTS ---")
        for i in session.get_inputs():
            print(f"Name: {i.name} | Shape: {i.shape} | Type: {i.type}")
            
        print("\n--- OUTPUTS ---")
        for o in session.get_outputs():
            print(f"Name: {o.name} | Shape: {o.shape} | Type: {o.type}")
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")

# Check both of your models
check_model("isidet/runs/segment/models/yolov26/23-03-20262/weights/best.onnx")
check_model("isidet/models/rfdetr/24-03-2026_0120/inference_model.onnx")
