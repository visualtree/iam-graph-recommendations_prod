#!/usr/bin/env python3
"""
Setup script for Streamlit modular structure
Creates directories and empty module files
"""

import os
from pathlib import Path

def create_streamlit_structure():
    """Create the modular Streamlit directory structure with all module files"""
    
    print("IAM Access Prediction - Streamlit Structure Setup")
    print("=" * 55)
    
    # Create main streamlit_modules directory
    modules_dir = Path("streamlit_modules")
    modules_dir.mkdir(exist_ok=True)
    print("Created streamlit_modules/ directory")
    
    # Create __init__.py for the package
    init_file = modules_dir / "__init__.py"
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write('"""Streamlit modules for IAM Access Prediction Demo"""\n')
    print("Created streamlit_modules/__init__.py")
    
    # List of all module files to create
    module_files = [
        "config.py",
        "session_management.py", 
        "data_loader.py",
        "ui_components.py",
        "metrics_calculator.py",
        "prediction_engine.py",
        "results_display.py",
        "explainability.py",
        "data_overview.py",
        "analysis_modules.py"
    ]
    
    # Create empty module files with basic docstrings
    for module_name in module_files:
        module_file = modules_dir / module_name
        if not module_file.exists():
            with open(module_file, 'w', encoding='utf-8') as f:
                f.write(f'"""\n{module_name.replace(".py", "").replace("_", " ").title()} module for Streamlit application\n"""\n\n')
                f.write('import streamlit as st\n\n')
                f.write('# TODO: Implement module functionality\n')
            print(f"Created streamlit_modules/{module_name}")
        else:
            print(f"streamlit_modules/{module_name} already exists")
    
    # Create main.py in project root
    main_file = Path("main.py")
    if not main_file.exists():
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('"""\nMain Streamlit Application for IAM Access Prediction Demo\n"""\n\n')
            f.write('import streamlit as st\n\n')
            f.write('# TODO: Import modules from streamlit_modules\n')
            f.write('# Example:\n')
            f.write('# from streamlit_modules.config import setup_page_config\n')
            f.write('# from streamlit_modules.ui_components import display_header\n\n')
            f.write('def main():\n')
            f.write('    """Main application entry point"""\n')
            f.write('    st.title("IAM Access Prediction Engine")\n')
            f.write('    st.write("Welcome to the demo!")\n')
            f.write('    \n')
            f.write('    # TODO: Implement main application logic\n\n')
            f.write('if __name__ == "__main__":\n')
            f.write('    main()\n')
        print("Created main.py")
    else:
        print("main.py already exists")
    
    # Create .streamlit config directory
    streamlit_config_dir = Path(".streamlit")
    streamlit_config_dir.mkdir(exist_ok=True)
    print("Created .streamlit/ directory")
    
    # Create config.toml
    config_toml = streamlit_config_dir / "config.toml"
    if not config_toml.exists():
        with open(config_toml, 'w', encoding='utf-8') as f:
            f.write('[global]\n')
            f.write('developmentMode = false\n\n')
            f.write('[server]\n')
            f.write('port = 8501\n')
            f.write('enableCORS = false\n')
            f.write('enableXsrfProtection = false\n')
            f.write('maxUploadSize = 1000\n\n')
            f.write('[browser]\n')
            f.write('gatherUsageStats = false\n\n')
            f.write('[theme]\n')
            f.write('primaryColor = "#1f77b4"\n')
            f.write('backgroundColor = "#ffffff"\n')
            f.write('secondaryBackgroundColor = "#f0f2f6"\n')
            f.write('textColor = "#262730"\n')
            f.write('font = "sans serif"\n')
        print("Created .streamlit/config.toml")
    else:
        print(".streamlit/config.toml already exists")
    
    # Create requirements file for demo
    requirements_file = Path("requirements_streamlit_demo.txt")
    if not requirements_file.exists():
        with open(requirements_file, 'w', encoding='utf-8') as f:
            f.write('# Streamlit Demo Requirements\n')
            f.write('streamlit>=1.28.0\n')
            f.write('plotly>=5.15.0\n')
            f.write('shap>=0.42.0\n')
            f.write('psutil>=5.9.0\n\n')
            f.write('# Data manipulation (if not already installed)\n')
            f.write('pandas>=1.5.0\n')
            f.write('numpy>=1.21.0\n\n')
            f.write('# ML libraries (if not already installed)\n')
            f.write('scikit-learn>=1.3.0\n')
            f.write('xgboost>=1.7.0\n')
            f.write('joblib>=1.3.0\n\n')
            f.write('# Neo4j connector (if not already installed)\n')
            f.write('neo4j>=5.8.0\n')
        print("Created requirements_streamlit_demo.txt")
    else:
        print("requirements_streamlit_demo.txt already exists")
    
    # Create launch script
    launcher_file = Path("launch_demo.py")
    if not launcher_file.exists():
        with open(launcher_file, 'w', encoding='utf-8') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('# Simple launcher for IAM Access Prediction Demo\n\n')
            f.write('import subprocess\n')
            f.write('import sys\n')
            f.write('import os\n\n')
            f.write('def main():\n')
            f.write('    print("IAM Access Prediction Demo Launcher")\n')
            f.write('    print("=" * 50)\n')
            f.write('    \n')
            f.write('    if not os.path.exists("main.py"):\n')
            f.write('        print("main.py not found. Please run from the project root directory.")\n')
            f.write('        return\n')
            f.write('    \n')
            f.write('    if not os.path.exists("streamlit_modules"):\n')
            f.write('        print("streamlit_modules directory not found.")\n')
            f.write('        return\n')
            f.write('    \n')
            f.write('    print("Launching Streamlit demo...")\n')
            f.write('    \n')
            f.write('    try:\n')
            f.write('        subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"], check=True)\n')
            f.write('    except subprocess.CalledProcessError as e:\n')
            f.write('        print(f"Failed to launch Streamlit: {e}")\n')
            f.write('    except KeyboardInterrupt:\n')
            f.write('        print("\\nDemo stopped by user")\n\n')
            f.write('if __name__ == "__main__":\n')
            f.write('    main()\n')
        print("Created launch_demo.py")
    else:
        print("launch_demo.py already exists")
    
    print("\nComplete structure created successfully!")
    print("\nCreated:")
    print("streamlit_modules/")
    print("   __init__.py")
    for module in module_files:
        print(f"   {module}")
    print(".streamlit/")
    print("   config.toml")
    print("main.py")
    print("requirements_streamlit_demo.txt")
    print("launch_demo.py")
    
    print(f"\nSummary:")
    print(f"   {len(module_files)} module files created")
    print(f"   1 main application file")
    print(f"   1 configuration directory")
    print(f"   2 setup files")
    
    print("\nNext steps:")
    print("1. Copy the actual module code into each file in streamlit_modules/")
    print("2. Copy the actual main.py code to replace the placeholder")
    print("3. Install requirements: pip install -r requirements_streamlit_demo.txt")
    print("4. Run: python launch_demo.py")

if __name__ == "__main__":
    create_streamlit_structure()
    print("\nSetup complete! All module files created and ready for code.")