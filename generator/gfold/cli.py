import argparse
import os

from .config import GFoldConfig
from .solver import GFoldSolver
from .visualization import plot_results

def main():
    """Command-line interface for the G-FOLD solver."""
    parser = argparse.ArgumentParser(description="G-FOLD: Fuel Optimal Large Divert Guidance Algorithm")
    parser.add_argument('-n', type=int, 
                        help="Determines the amount of steps and thus determines dt (dt = tf/N)", 
                        default=100)
    parser.add_argument('-g', '--generate', 
                        help="Generate C++/Python code to solve problems faster", 
                        action="store_true")
    parser.add_argument('-o', '--output', 
                        help="Output directory for generated code or plot", 
                        default=".")
    parser.add_argument('--no-plot', 
                        help="Don't display the plot", 
                        action="store_true")
    parser.add_argument('-s', '--save-plot', 
                        help="Save the plot to a file (default: don't save)", 
                        action="store_true")
    
    args = parser.parse_args()
    
    # Initialize solver
    solver = GFoldSolver(GFoldConfig(n=args.n))
    
    # Generate code if requested
    if args.generate:
        output_dir = os.path.join(args.output, "code")
        code_dir = solver.generate_code(code_dir=output_dir)
        print(f"Generated code in {code_dir}")
        return
    
    # Otherwise solve and visualize
    print("Solving G-FOLD optimization problem...")
    solution = solver.solve(verbose=True)
    print(f"Final mass: {solution['final_mass']:.2f} kg")
    
    # Plot results
    save_path = os.path.join(args.output, "gfold_plot.png") if args.save_plot else None
    plot_results(solution, save_path=save_path, show=not args.no_plot)

if __name__ == "__main__":
    main()
