import numpy as np
import json
from scipy.optimize import linear_sum_assignment

def calculate_path_time(path_length_feet, num_nodes, is_drone=True):
    """
    Calculate total time for an agent to traverse a path.
    
    Args:
        path_length_feet: Total path length in feet
        num_nodes: Number of nodes in the path
        is_drone: Boolean indicating if the agent is a drone (True) or firefighter (False)
    
    Returns:
        Total time in seconds
    """
    # Convert path length from feet to miles
    path_length_miles = path_length_feet / 5280.0
    
    # Speed in miles per hour
    speed_mph = 30 if is_drone else 10  # 30 mph for drone, 10 mph for firefighter
    speed_mps = speed_mph * 0.44704  # Convert mph to meters per second
    
    # Convert path length from miles to meters
    path_length_meters = path_length_miles * 1609.34
    
    # Calculate movement time in seconds
    movement_time = path_length_meters / speed_mps
    
    # Calculate waiting time at each node
    wait_time_per_node = 10 if is_drone else 20  # seconds
    total_wait_time = (num_nodes - 1) * wait_time_per_node  # No wait at the last node
    
    return movement_time + total_wait_time

def assign_agents(paths, agent_types):
    """
    Assign agents to paths using the Hungarian algorithm.
    
    Args:
        paths: List of dictionaries, each containing:
               - 'length': path length in miles
               - 'nodes': list of node indices in the path
        agent_types: List of agent types ('drone' or 'firefighter')
    
    Returns:
        A dictionary containing:
        - 'assignments': List of (agent_idx, path_idx) tuples
        - 'total_time': Total time for all assignments
        - 'agent_times': List of times for each agent
    """
    num_agents = len(agent_types)
    num_paths = len(paths)
    
    # Create cost matrix (time matrix)
    cost_matrix = np.zeros((num_agents, num_paths))
    
    for i, agent_type in enumerate(agent_types):
        is_drone = (agent_type.lower() == 'drone')
        for j, path in enumerate(paths):
            path_length = path['length']
            num_nodes = len(path['nodes'])
            cost_matrix[i, j] = calculate_path_time(path_length, num_nodes, is_drone)
    
    # Use Hungarian algorithm to find optimal assignment
    agent_indices, path_indices = linear_sum_assignment(cost_matrix)
    
    # Prepare results
    assignments = []
    agent_times = []
    total_time = 0
    
    for agent_idx, path_idx in zip(agent_indices, path_indices):
        time_taken = cost_matrix[agent_idx, path_idx]
        assignments.append({
            'agent_idx': agent_idx,
            'agent_type': agent_types[agent_idx],
            'path_idx': path_idx,
            'time_seconds': time_taken,
            'path_nodes': paths[path_idx]['nodes']
        })
        agent_times.append(time_taken)
        total_time += time_taken
    
    return {
        'assignments': assignments,
        'total_time': total_time,
        'agent_times': agent_times,
        'cost_matrix': cost_matrix
    }

def get_paths_from_customsolution(points, num_agents=5, time_limit=10):
    """
    Get paths from customsolution.py and convert to format for Hungarian algorithm.
    
    Args:
        points: List of [x,y] coordinates (first point is the depot)
        num_agents: Number of agents/salesmen
        time_limit: Time limit for the solver in seconds
        
    Returns:
        List of path dictionaries with 'length' and 'nodes' keys
    """
    # Get solution from customsolution
    solution = solve_custom_mtsp(points, num_agents=num_agents, time_limit=time_limit)
    
    # Convert to path format expected by Hungarian algorithm
    paths = []
    for route in solution['routes']:
        # Calculate path length in miles (convert from meters)
        path_length = 0
        for i in range(len(route) - 1):
            # Get coordinates of current and next point
            p1 = solution['points'][route[i]]
            p2 = solution['points'][route[i + 1]]
            # Calculate Euclidean distance and convert to miles
            distance_meters = np.linalg.norm(p1 - p2)
            distance_miles = distance_meters / 1609.34  # Convert meters to miles
            path_length += distance_miles
            
        paths.append({
            'length': path_length,
            'nodes': route.tolist()  # Convert numpy array to list for JSON serialization
        })
    
    return paths

def calculate_path_length(points, path):
    """
    Calculate the total length of a path in feet, excluding movement between stairwell points.
    Stairwell indices are 1 and 2 as per the problem definition.
    """
    total_length = 0
    stairwell_indices = {1, 2}  # Stairwell points
    
    i = 0
    while i < len(path) - 1:
        current_node = path[i]
        next_node = path[i+1]
        
        # Only add distance if neither current nor next node is a stairwell
        if current_node not in stairwell_indices and next_node not in stairwell_indices:
            x1, y1 = points[current_node]
            x2, y2 = points[next_node]
            # Calculate Euclidean distance in feet
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            total_length += distance
        
        i += 1
    
    return total_length

def read_paths_from_file(filename='solutionpaths.json'):
    """
    Read paths from the solutionpaths.json file.
    
    Args:
        filename: Name of the JSON file to read from
        
    Returns:
        List of path dictionaries with 'length' and 'nodes' keys
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Define the points (same as in customsolution2.py)
    points = [
        [0, 0], [72, 0], [36, 0], [6, 3], [18, 3], [30, 3], 
        [42, 3], [54, 3], [66, 3], 
        [78, 3], [90, 3], [102, 3],
        [6, -3], [18, -3], [30, -3],
        [42, -3], [54, -3], [66, -3],
        [78, -3], [90, -3], [102, -3]
    ]

    stairwell_indices = [1]

    
    paths = []
    for path_name, path_nodes in data.items():
        if path_name.startswith('path'):
            # Calculate the actual path length
            path_length = calculate_path_length(points, path_nodes)
            paths.append({
                'length': path_length,
                'nodes': path_nodes
            })
    
    return paths

def print_assignment(assignment_result, paths):
    """Print the assignment results in a readable format."""
    print("\n=== Optimal Assignment ===")
    print(f"Total time for all assignments: {assignment_result['total_time']:.2f} seconds")
    
    max_time = 0
    last_agent = None
    
    print("\nIndividual Assignments:")
    for i, assign in enumerate(assignment_result['assignments']):
        print(f"Agent {i+1} ({assign['agent_type']}):")
        print(f"  - Assigned to path: {assign['path_nodes']}")
        print(f"  - Time taken: {assign['time_seconds']:.2f} seconds")
        print(f"  - Path length: {paths[assign['path_idx']]['length']:.2f} feet")
        print()
        
        # Track the agent with the longest time
        if assign['time_seconds'] > max_time:
            max_time = assign['time_seconds']
            last_agent = f"Agent {i+1} ({assign['agent_type']})"
    
    print(f"\nLast agent to finish: {last_agent} with {max_time:.2f} seconds")

# Example usage
if __name__ == "__main__":
    try:
        # Read paths from the JSON file
        paths = read_paths_from_file('solutionpaths.json')
        
        if not paths:
            print("No valid paths found in the solution file.")
        else:
            # Create a balanced number of agent types based on number of paths
            num_paths = len(paths)
            num_firefighters = 2#(num_paths + 1) // 2  # At least half firefighters
            agent_types = (['firefighter'] * num_firefighters + 
                          ['drone'] * (num_paths - num_firefighters))
            
            print(f"Found {num_paths} paths in the solution.")
            print(f"Agent distribution: {num_firefighters} firefighters, "
                  f"{num_paths - num_firefighters} drones")
            
            # Get optimal assignment
            result = assign_agents(paths, agent_types)
            
            # Print results
            print_assignment(result, paths)
            
            # Print the cost matrix for reference
            print("\nCost Matrix (time in seconds):")
            print("Rows: Agents, Columns: Paths")
            print("Agent types:", agent_types)
            print(result['cost_matrix'].round(2))
        
    except FileNotFoundError:
        print("Error: solutionpaths.json not found. Please run customsolution2.py first.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()