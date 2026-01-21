#!/usr/bin/env python3
import os
import subprocess
import sys

# Shader directory configuration
SHADER_DIR = "dependencies/BTQ_Render_Engine/shaders"
SPIRV_DIR = os.path.join(SHADER_DIR, "spirv")

def main():
    # Create SPIR-V output directory if it doesn't exist
    os.makedirs(SPIRV_DIR, exist_ok=True)
    
    # Find all shader files
    shader_extensions = [".comp", ".vert", ".frag"]
    shader_files = []
    
    for filename in os.listdir(SHADER_DIR):
        if any(filename.endswith(ext) for ext in shader_extensions):
            shader_files.append(os.path.join(SHADER_DIR, filename))
    
    if not shader_files:
        print("No shader files found in", SHADER_DIR)
        return 0
    
    print(f"Found {len(shader_files)} shader file(s)")
    
    # Compile each shader
    successful = 0
    failed = 0
    
    for shader_path in shader_files:
        filename = os.path.basename(shader_path)
        name, ext = os.path.splitext(filename)
        
        # Determine output SPIR-V file path
        output_path = os.path.join(SPIRV_DIR, f"{name}.spv")
        
        try:
            print(f"Compiling {filename}...")
            
            # Run glslc compiler
            result = subprocess.run(
                ["glslc", shader_path, "-o", output_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout:
                print(f"Output: {result.stdout.strip()}")
            if result.stderr:
                print(f"Warnings: {result.stderr.strip()}")
                
            print(f"Successfully compiled to {os.path.basename(output_path)}")
            successful += 1
            
        except FileNotFoundError:
            print("ERROR: glslc compiler not found. Please install Vulkan SDK.")
            return 1
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Compilation failed for {filename}:")
            print(e.stdout)
            print(e.stderr)
            failed += 1
        except Exception as e:
            print(f"ERROR: Unexpected error compiling {filename}: {e}")
            failed += 1
    
    # Summary
    print("\nCompilation Summary:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("All shaders compiled successfully!")
        return 0
    else:
        print(f"Some shaders failed to compile ({failed} out of {successful + failed})")
        return 1

if __name__ == "__main__":
    sys.exit(main())
