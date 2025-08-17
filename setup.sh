#!/usr/bin/env bash

# Run a local Chroma database
docker run -p 8000:8000 chromadb/chroma

# Get the data
wget -O ./data/FoodDataSet.cjs https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/CBCVVX4wJjXG64DKYMVi1w/FoodDataSet.js
wget -P ./data https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/HoMe0o66TlJJ-WrIcR_8HQ/Chocolate-torte-Recipe.pdf
wget -P ./data https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GvUxpXUD-oy1h5z-qKoVFg/crumble-pie.pdf