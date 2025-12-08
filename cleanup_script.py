#!/usr/bin/env python
"""
Cleanup script for the Physical-AI-and-Humanoid-Robotics project
This script removes redundant files and consolidates the project structure
"""

import os
import shutil
from pathlib import Path

def main():
    project_root = Path("D:\\AI\\Hackathon-I\\Physical-AI-and-Humanoid-Robotics")
    
    print("Starting project cleanup...")
    
    # 1. Remove duplicate backend directory (keep the more complete one)
    rag_backend_dir = project_root / "rag-chatbot-backend"
    if rag_backend_dir.exists():
        print(f"Removing duplicate backend: {rag_backend_dir}")
        shutil.rmtree(rag_backend_dir)
    
    # 2. Remove redundant root files
    root_files_to_remove = [
        "main.py",  # Duplicate of backend entry point
        "sample_robotics_content.txt",  # Should be in docs
        "static",  # Conflicts with Docusaurus
        "src"  # Conflicts with Docusaurus
    ]
    
    for file in root_files_to_remove:
        file_path = project_root / file
        if file_path.exists():
            print(f"Removing redundant root file/directory: {file_path}")
            if file_path.is_file():
                file_path.unlink()
            else:
                shutil.rmtree(file_path)
    
    # 3. Update environment.yml to match backend Python version
    env_file = project_root / "environment.yml"
    if env_file.exists():
        content = env_file.read_text()
        # Replace Python 3.9 with 3.11 to match backend requirements
        content = content.replace("python=3.9", "python=3.11")
        env_file.write_text(content)
        print("Updated environment.yml to use Python 3.11")
    
    # 4. Remove duplicate vercel.json from backend (should only be in frontend)
    backend_vercel = project_root / "backend" / "vercel.json"
    if backend_vercel.exists():
        backend_vercel.unlink()
        print("Removed duplicate vercel.json from backend directory")
    
    print("Cleanup completed!")
    print("\nKey changes made:")
    print("1. Removed duplicate rag-chatbot-backend directory")
    print("2. Removed conflicting src/ and static/ directories from root")
    print("3. Removed redundant main.py from root")
    print("4. Updated environment.yml to Python 3.11")
    print("5. Removed duplicate vercel.json from backend")
    print("\nThe project now has a cleaner, more focused structure.")

if __name__ == "__main__":
    main()