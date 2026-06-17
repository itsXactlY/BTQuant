#!/usr/bin/env python3
import os
import subprocess
import sys

BTQ_PYTHON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", ".btq", "bin", "python"
)

def main():
    python = BTQ_PYTHON if os.path.isfile(BTQ_PYTHON) else sys.executable
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantstats_dashboard.py")

    try:
        subprocess.run(
            [python, "-m", "streamlit", "run", script],
            check=True,
            cwd=os.path.dirname(script),
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"Streamlit exited with an error: {e}")

if __name__ == "__main__":
    main()
