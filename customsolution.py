import torch, json
import numpy as np
from ortools_mtsp import my_solve_mtsp
import matplotlib.pyplot as plt

def solve_custom_mtsp(points, num_agents=5, time_limit=10):
    """
    Solve MTSP for a custom set of points.
    
    Args:
        points: List of [x,y] coordinates (first point is the depot)
        num_agents: Number of salesmen
        time_limit: Time limit for OR-Tools solver in seconds
    """
    # Convert points to numpy array
    points = np.array(points, dtype=np.float32)
    
    # Calculate distance matrix (Euclidean distance)
    n = len(points)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])
    
    # Scale up for better OR-Tools precision
    dist_matrix = (dist_matrix * 1000).astype(np.int64)
    
    # Solve MTSP
    max_length, routes, all_lengths = my_solve_mtsp(dist_matrix, num_agents, time_limit)
    
    return {
        'max_route_length': max_length,
        'routes': routes,
        'route_lengths': all_lengths,
        'points': points
    }

def plot_solution(solution):
    """Visualize the MTSP solution"""
    points = solution['points']
    routes = solution['routes']
    
    plt.figure(figsize=(10, 8))
    
    # Plot all points
    plt.scatter(points[:, 0], points[:, 1], c='red', s=100, zorder=5)
    
    # Highlight depot
    plt.scatter(points[0, 0], points[0, 1], c='green', s=200, marker='*', zorder=10)
    
    # Plot routes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
    for i, (route, color) in enumerate(zip(routes, colors)):
        route_points = points[route]
        plt.plot(route_points[:, 0], route_points[:, 1], 'o-', color=color, 
                label=f'Agent {i+1}: {solution["route_lengths"][i]:.2f}')
    
    plt.title(f'MTSP Solution (Max route: {solution["max_route_length"]:.2f})')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def solve_flexible_mtsp(points, start_points, end_points=None, time_limit=10, return_to_start=False):
    """
    Solve MTSP with flexible start/end points for multiple agents.
    
    Args:
        points: List of [x,y] coordinates of all locations
        start_points: List of indices in 'points' for each agent's start location
        end_points: Optional list of indices for each agent's end location.
                   If None, agents return to their start points.
                   If -1, agent can end anywhere.
        time_limit: Time limit for OR-Tools in seconds
        return_to_start: If True, forces all agents to return to their start points
    """
    import numpy as np
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp

    # Calculate distance matrix
    n = len(points)
    dist_matrix = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = int(np.linalg.norm(
                np.array(points[i]) - np.array(points[j])
            ) * 1000)  # Scale for precision

    num_vehicles = len(start_points)
    
    # Handle end points
    if end_points is None:
        end_points = start_points  # Default: return to start
    elif isinstance(end_points, int) and end_points == -1:
        end_points = [-1] * num_vehicles  # Can end anywhere
    elif len(end_points) != num_vehicles:
        raise ValueError("end_points must have same length as start_points or be -1")

    # Create routing index manager with multiple depots
    manager = pywrapcp.RoutingIndexManager(
        n,  # number of locations
        num_vehicles,  # number of vehicles
        start_points,  # start depots
        end_points    # end depots (can be same as start or different)
    )

    # Create routing model
    routing = pywrapcp.RoutingModel(manager)

    # Register distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add distance dimension
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        300000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # If return_to_start is True, force return to start
    if return_to_start:
        for vehicle_id in range(num_vehicles):
            end_node = manager.NodeToIndex(start_points[vehicle_id])
            routing.solver().Add(
                routing.NextVar(routing.End(vehicle_id)) == end_node
            )

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = time_limit

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Extract solution
    if solution:
        routes = []
        route_lengths = []
        max_route_distance = 0
        used_starts = set()
        
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            
            # Add the end node
            node = manager.IndexToNode(index)
            route.append(node)
            routes.append(route)
            route_lengths.append(route_distance / 1000.0)
            max_route_distance = max(max_route_distance, route_distance / 1000.0)
            used_starts.add(route[0])  # Track which starts were actually used
        
        # Find unused starts
        all_starts = set(start_points)
        unused_starts = all_starts - used_starts
        
        return {
            'max_route_length': max_route_distance,
            'routes': routes,
            'route_lengths': route_lengths,
            'points': points,
            'unused_starts': unused_starts
        }
    return None

def plot_solution(solution):
    """Visualize the MTSP solution with multiple agents"""
    points = solution['points']
    routes = solution['routes']
    
    plt.figure(figsize=(12, 10))
    
    # Plot all points
    points_array = np.array(points)
    plt.scatter(points_array[:, 0], points_array[:, 1], c='gray', s=100, alpha=0.5)
    
    # Plot routes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
    for i, (route, color) in enumerate(zip(routes, colors)):
        if not route:
            continue
        route_points = points_array[route]
        plt.plot(route_points[:, 0], route_points[:, 1], 'o-', color=color, 
                linewidth=2, markersize=8, 
                label=f'Agent {i+1}: {solution["route_lengths"][i]:.2f}')
        
        # Mark start and end points
        start = route[0]
        end = route[-1]
        plt.scatter(points_array[start, 0], points_array[start, 1], 
                   c='green', s=200, marker='*', edgecolor='black', zorder=10)
        if start != end:
            plt.scatter(points_array[end, 0], points_array[end, 1], 
                       c='red', s=200, marker='s', edgecolor='black', zorder=10)
    
    # Add legend and title
    plt.legend(loc='upper right')
    plt.title(f'MTSP Solution (Max route: {solution["max_route_length"]:.2f})')
    plt.grid(True)
    plt.axis('equal')
    
    # Add unused starts info
    if 'unused_starts' in solution and solution['unused_starts']:
        unused_points = [points[i] for i in solution['unused_starts']]
        unused_x, unused_y = zip(*unused_points)
        plt.scatter(unused_x, unused_y, c='black', s=300, marker='x', 
                   linewidth=2, label='Unused Starts')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def export_solution(solution, filename='solutionpaths.json'):
    """
    Export the solution to a JSON file in the required format.
    """
    if not solution:
        print("No solution to export.")
        return
    
    # Convert the solution to the required format
    paths = {}
    path_counter = 1
    for i, route in enumerate(solution['routes']):
        if len(route) > 2:  # Only include non-empty paths (more than just [0, 0])
            paths[f'path{path_counter}'] = route
            path_counter += 1
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(paths, f, indent=2)
    
    print(f"\nSolution exported to {filename}")
    print("-" * 50)
    print(json.dumps(paths, indent=2))
    print("-" * 50)

if __name__ == "__main__":
    # Example usage with 3 agents and flexible start/end points
    points = [
        [0, 0],      # Exit left
        [36, 0],     # Exit right
        [6, 3],      # Top left door
        [18, 3],     # Top middle door
        [30, 3],     # Top right door
        [30, -3],    # Bottom right door
        [18, -3],    # Bottom middle door
        [6, -3]      # Bottom left door
    ]

    points2 = [
        [0, 0], [72, 0], [36, 0], [6, 3], [18, 3], [30, 3], 
        [42, 3], [54, 3], [66, 3], 
        [78, 3], [90, 3], [102, 3],
        [6, -3], [18, -3], [30, -3],
        [42, -3], [54, -3], [66, -3],
        [78, -3], [90, -3], [102, -3]
    ]

    # Each agent has a start point, can end at any point
    start_points = [0,1,0,1]  # Agents start at points 1, 2, and 3
    stairwell_points = [1, 2]
    end_points = [0,1,0,1]  # -1 means can end anywhere

    # Solve with flexible end points
    solution = solve_flexible_mtsp(
        points=points,
        start_points=start_points,
        end_points=end_points,
        time_limit=10,
        return_to_start=False  # Set to True to force return to start
    )

    # Print and plot solution
    if solution:
        print(f"Path lengths: {[f'{l:.2f}' for l in solution['route_lengths']]}")
        print(f"Maximum route length: {solution['max_route_length']:.2f}")
        
        # Print route information
        for i, route in enumerate(solution['routes']):
            if len(route) > 2:  # Only show non-empty routes
                print(f"Agent {i+1} route: {route} (length: {solution['route_lengths'][i]:.2f})")
                print(f"  Starts at: {points[route[0]]}")
                print(f"  Ends at: {points[route[-1]]}\n")
        
        # Export solution in the required format
        export_solution(solution, 'solutionpaths.json')
        
        # Also save the full solution data for reference
        solution_data = {
            'routes': solution['routes'],
            'route_lengths': solution['route_lengths'],
            'points': solution['points'].tolist() if hasattr(solution['points'], 'tolist') else solution['points']
        }
        with open('solution_paths.json', 'w') as f:
            json.dump(solution_data, f)
        print("\nSolution data saved to 'solution_paths.json'")
        
        # Show the visualization
        plot_solution(solution)