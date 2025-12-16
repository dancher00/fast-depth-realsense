"""
Entry-point helper to run the full project:
- benchmark
- visualizations
- profiling
"""
import sys
import os

def main():
    print("="*60)
    print("HIGHPERFORMANCE REALSENSE DEPTH PROCESSING")
    print("Final project for the 'High‑Performance Python' course")
    print("="*60)
    
    print("\nAvailable options:")
    print("1. Run benchmark (generates all plots)")
    print("2. Create visualizations")
    print("3. Profile code")
    print("4. Run everything (profiling + benchmark + visualizations)")
    
    choice = input("\nChoose an option (1-4): ").strip()
    
    if choice == '1':
        print("\nRunning benchmark...")
        os.system("python benchmark.py")
    
    elif choice == '2':
        print("\nCreating visualizations...")
        os.system("python visualize_results.py")
    
    elif choice == '3':
        print("\nProfiling code...")
        os.system("python profile_code.py")
    
    elif choice == '4':
        print("\nRunning the full pipeline...")
        print("\n1/3: Profiling...")
        os.system("python profile_code.py")
        print("\n2/3: Benchmark...")
        os.system("python benchmark.py")
        print("\n3/3: Visualizations...")
        os.system("python visualize_results.py")
        print("\n✅ All tasks completed! See 'plots/' for figures.")
    
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)

