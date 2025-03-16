"""
Simple example demonstrating the use of the G-FOLD solver.
"""
from gfold import GFoldSolver
from gfold.visualization import plot_results

def main():
    # Create solver with default configuration
    solver = GFoldSolver()
    
    # Solve the problem
    solution = solver.solve(verbose=True)
    
    # Print results
    print(f"Final mass: {solution['final_mass']:.2f} kg")
    
    # Plot and save results
    plot_results(solution, save_path="example_plot.png")
    
    print("Example completed. Plot saved to example_plot.png")

if __name__ == "__main__":
    main()
