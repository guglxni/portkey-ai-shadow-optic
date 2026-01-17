#!/usr/bin/env python3
"""
Phoenix Server with Persistent Storage
=======================================
Starts Phoenix with SQLite database for persistent trace storage.
Run this as a separate process before running the demo.
"""

import os
import sys
from pathlib import Path

# Set up persistent storage BEFORE importing phoenix
PHOENIX_DIR = Path(__file__).parent.parent / ".phoenix_data"
PHOENIX_DIR.mkdir(exist_ok=True)

# Configure Phoenix to use SQLite for persistence
os.environ["PHOENIX_WORKING_DIR"] = str(PHOENIX_DIR)
os.environ["PHOENIX_SQL_DATABASE_URL"] = f"sqlite:///{PHOENIX_DIR}/phoenix.db"

# Clear any conflicting environment variables
os.environ.pop("PHOENIX_COLLECTOR_ENDPOINT", None)

import phoenix as px

def main():
    print("🔮 Starting Phoenix with Persistent Storage...")
    print(f"   Database: {PHOENIX_DIR}/phoenix.db")
    
    # Launch Phoenix with explicit configuration
    session = px.launch_app()
    
    if session:
        print(f"\n✅ Phoenix UI running at: {session.url}")
        print(f"   GRPC Collector: localhost:4317")
        print(f"   HTTP Collector: localhost:6006/v1/traces")
        print("\n📊 Traces will be stored in: .phoenix_data/phoenix.db")
        print("\nPress Ctrl+C to stop Phoenix server...")
        
        # Keep the server running
        import time
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n👋 Phoenix server stopped.")
    else:
        print("❌ Failed to start Phoenix")
        sys.exit(1)

if __name__ == "__main__":
    main()
