import os
import sys

# Create shaders directory structure
shader_dir = "dependencies/BTQ_Render_Engine/shaders"
spirv_dir = os.path.join(shader_dir, "spirv")

if not os.path.exists(shader_dir):
    os.makedirs(shader_dir)
    print(f"Created directory: {shader_dir}")

if not os.path.exists(spirv_dir):
    os.makedirs(spirv_dir)
    print(f"Created directory: {spirv_dir}")

print("Shader directory structure created successfully")
