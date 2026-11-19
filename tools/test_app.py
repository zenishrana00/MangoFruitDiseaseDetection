import sys
import os
import io
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))
from app import app, CLASS_NAMES

client = app.test_client()

print("==================================================")
print("1. Testing Health Endpoint")
print("==================================================")
res = client.get("/health")
print("Health response:", res.status_code, res.get_json())
assert res.status_code == 200
assert res.get_json()["input_resolution"] == "224x224x3"

print("\n==================================================")
print("2. Testing Template Rendering with Camera & Upload Tabs")
print("==================================================")
res = client.get("/")
print("Index status code:", res.status_code)
assert res.status_code == 200
assert b"Upload from Device" in res.data
assert b"Capture from Camera" in res.data
assert b"cameraVideo" in res.data
assert b"Image optimized for AI analysis" in res.data
print("Template rendered successfully with Camera and Upload options!")

print("\n==================================================")
print("3. Testing Error Handling (Invalid & Missing Inputs)")
print("==================================================")
res = client.post("/predict", data={})
assert res.status_code == 400

txt_data = {"image": (io.BytesIO(b"Hello world, not an image"), "sample.txt")}
res = client.post("/predict", data=txt_data, content_type="multipart/form-data")
assert res.status_code == 400
assert "Unsupported file format" in res.get_json()["error"]
print("Validation error handling passed!")

print("\n==================================================")
print("4. Testing Different Resolutions & Aspect Ratios")
print("==================================================")
# Test Landscape Image (1920 x 1080)
landscape_img = Image.new("RGB", (1920, 1080), color=(120, 180, 50))
b_land = io.BytesIO()
landscape_img.save(b_land, format="JPEG")
b_land.seek(0)
res = client.post("/predict", data={"image": (b_land, "landscape.jpg")}, content_type="multipart/form-data")
assert res.status_code == 200
data_land = res.get_json()
print(f"Landscape 1920x1080: Orig={data_land['original_resolution']} -> {data_land['optimized_resolution']} | {data_land['optimization_message']}")
assert data_land["original_resolution"] == "1920 × 1080"
assert data_land["optimized_resolution"] == "224 × 224 × 3"

# Test Portrait Image (1080 x 1920 - typical mobile camera capture)
portrait_img = Image.new("RGB", (1080, 1920), color=(220, 150, 40))
b_port = io.BytesIO()
portrait_img.save(b_port, format="JPEG")
b_port.seek(0)
res = client.post("/predict", data={"image": (b_port, "camera_mango_capture.jpg")}, content_type="multipart/form-data")
assert res.status_code == 200
data_port = res.get_json()
print(f"Portrait 1080x1920 (Camera Capture): Orig={data_port['original_resolution']} -> {data_port['optimized_resolution']} | {data_port['optimization_message']}")
assert data_port["original_resolution"] == "1080 × 1920"

print("\n==================================================")
print("5. Testing Prediction on All 7 Real Disease Classes")
print("==================================================")
test_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "final", "test"))

for class_name in CLASS_NAMES:
    cat_dir = os.path.join(test_base, class_name)
    files = [f for f in os.listdir(cat_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        continue

    sample_img_path = os.path.join(cat_dir, files[0])
    with open(sample_img_path, "rb") as img_file:
        img_bytes = img_file.read()

    post_data = {"image": (io.BytesIO(img_bytes), files[0])}
    res = client.post("/predict", data=post_data, content_type="multipart/form-data")
    assert res.status_code == 200
    
    result = res.get_json()
    assert result["success"] is True
    print(f"Test Class: {class_name:<16} -> Predicted: {result['display_name']:<16} | Confidence: {result['confidence']:6.2f}% | Dims: {result['original_resolution']}")

print("\n==================================================")
print("ALL TESTS PASSED SUCCESSFULLY WITH 100% SUCCESS!")
print("==================================================")
