#!/usr/bin/env python3
import subprocess
import sys

def main():
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "quantstats_dashboard.py"],
            check=True
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"Streamlit exited with an error: {e}")

if __name__ == "__main__":
    main()
