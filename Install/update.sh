#!/bin/bash
source .btq/bin/activate
git pull || { echo "Failed to update repository"; exit 1; }
pip install --upgrade pip || { echo "Failed to upgrade pip"; exit 1; }
cd dependencies || { echo "Failed to enter dependencies directory"; exit 1; }
pip install .
cd ..
