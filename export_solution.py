import json

def export_solution(solution, filename='solutionpaths.json'):
    """
    Export the solution to a JSON file in the required format.
    """
    if not solution:
        print("No solution to export.")
        return
    
    # Convert the solution to the required format
    paths = {}
    for i, route in enumerate(solution['routes']):
        if len(route) > 2:  # Only include non-empty paths (more than just [0, 0])
            # Convert to 1-based indexing if needed
            paths[f'path{i+1}'] = [point + 1 for point in route]  # +1 if you need 1-based indexing
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(paths, f, indent=2)
    
    print(f"\nSolution exported to {filename}")
    print("-" * 50)
    print(json.dumps(paths, indent=2))
    print("-" * 50)

# At the end of the script, after getting the solution
if __name__ == "__main__":
    # Your existing code to get the solution...
    # ...
    
    if solution:
        print("\nSolution found!")
        print("-" * 50)
        for i, route in enumerate(solution['routes']):
            print(f"Agent {i+1} path: {route}")
        print("-" * 50)
        
        # Export the solution
        export_solution(solution)
        
        # Show the plot
        plot_solution(solution)
