import numpy as np
from scipy.optimize import linear_sum_assignment

def calculate_path_time(path_length_miles, num_nodes, is_drone=True):
    """
    Calculate total time for an agent to traverse a path.
    
    Args:
        path_length_miles: Total path length in miles
        num_nodes: Number of nodes in the path
        is_drone: Boolean indicating if the agent is a drone (True) or firefighter (False)
    
    Returns:
        Total time in seconds
    """
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

def print_assignment(assignment_result):
    """Print the assignment results in a readable format."""
    print("\n=== Optimal Assignment ===")
    print(f"Total time for all assignments: {assignment_result['total_time']:.2f} seconds\n")
    
    print("Individual Assignments:")
    for i, assign in enumerate(assignment_result['assignments']):
        print(f"Agent {i+1} ({assign['agent_type']}):")
        print(f"  - Assigned to path: {assign['path_nodes']}")
        print(f"  - Time taken: {assign['time_seconds']:.2f} seconds")
        print(f"  - Path length: {paths[assign['path_idx']]['length']:.2f} miles")
        print()

# Example usage
if __name__ == "__main__":
    # Example paths (from previous solution)
    paths = [
        {'length': 1.2, 'nodes': [0, 2, 4, 1]},  # Example path 1
        {'length': 0.8, 'nodes': [1, 5, 3]},      # Example path 2
        {'length': 1.5, 'nodes': [0, 6, 2, 5, 1]}, # Example path 3
        {'length': 1.0, 'nodes': [1, 7, 3, 0]}     # Example path 4
    ]
    
    # Example agent types
    agent_types = ['drone', 'firefighter', 'drone', 'firefighter']
    
    # Get optimal assignment
    result = assign_agents(paths, agent_types)
    
    # Print results
    print_assignment(result)
    
    # Print the cost matrix for reference
    print("\nCost Matrix (time in seconds):")
    print("Rows: Agents, Columns: Paths")
    print("Agent types:", agent_types)
    print(result['cost_matrix'].round(2))