from extract_thermal_features import extract_features

img = "data/raw/images/16551.jpg"

features = extract_features(img)

print(features)

# 6788 is for hotspot and 14249 for no anomaly