import torch
import numpy as np
import json
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import matplotlib.pyplot as plt

# --- KEEP YOUR EXISTING HELPER FUNCTIONS --- 
# (identify_floors and can_travel_directly are unchanged)
def identify_floors(points, stairwell_indices):
    stairwell_x_positions = set([points[i][0] for i in stairwell_indices])
    floor_assignments = {}
    if not stairwell_indices:
        for i in range(len(points)):
            floor_assignments[i] = 0
        return floor_assignments, stairwell_x_positions
    sorted_stairwell_x = sorted(stairwell_x_positions)
    for i, point in enumerate(points):
        if i in stairwell_indices:
            floor_assignments[i] = -1
        else:
            point_x = point[0]
            floor_num = 0
            for stair_x in sorted_stairwell_x:
                if point_x > stair_x:
                    floor_num += 1
                else:
                    break
            floor_assignments[i] = floor_num
    return floor_assignments, stairwell_x_positions

def solve_flexible_mtsp(points, start_points, end_points=None, stairwell_indices=None, 
                        time_limit=10, return_to_start=False):
    """
    Updated solver that forces load balancing among agents.
    """
    # Identify floors
    if stairwell_indices is None:
        stairwell_indices = []
    stairwell_set = set(stairwell_indices)
    
    floor_assignments, stairwell_x_positions = identify_floors(points, stairwell_indices)
    
    # Stairwell connection logic
    sorted_stairwell_x = sorted(stairwell_x_positions)
    stairwell_connections = {} 
    for stairwell_idx in stairwell_indices:
        stair_x = points[stairwell_idx][0]
        pos = sorted_stairwell_x.index(stair_x)
        stairwell_connections[stairwell_idx] = {pos, pos + 1}
    
    num_vehicles = len(start_points)
    original_n = len(points)
    copies_per_stairwell = 10 
    
    # Expand points array to include virtual stairwells
    expanded_points = list(points)
    virtual_stairwell_map = {} 
    
    for agent_id in range(num_vehicles):
        for stairwell_idx in stairwell_indices:
            for copy_num in range(copies_per_stairwell):
                virtual_idx = len(expanded_points)
                expanded_points.append(points[stairwell_idx]) 
                virtual_stairwell_map[(agent_id, stairwell_idx, copy_num)] = virtual_idx
    
    n = len(expanded_points)
    
    # Calculate base distance matrix
    base_dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            base_dist_matrix[i, j] = np.linalg.norm(
                np.array(expanded_points[i]) - np.array(expanded_points[j])
            )
    
    # Create modified distance matrix with floor restrictions
    dist_matrix = np.zeros((n, n), dtype=np.int64)
    LARGE_PENALTY = 10**9
    
    def get_original_idx(idx):
        if idx < original_n:
            return idx
        for (agent_id, orig_idx, copy_num), virtual_idx in virtual_stairwell_map.items():
            if virtual_idx == idx:
                return orig_idx
        return idx
    
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i, j] = 0
            else:
                orig_i = get_original_idx(i)
                orig_j = get_original_idx(j)
                
                i_is_stairwell = orig_i in stairwell_set
                j_is_stairwell = orig_j in stairwell_set
                
                can_travel = False
                
                if not i_is_stairwell and not j_is_stairwell:
                    if floor_assignments[orig_i] == floor_assignments[orig_j]:
                        can_travel = True
                elif i_is_stairwell and j_is_stairwell:
                    can_travel = True
                elif i_is_stairwell and not j_is_stairwell:
                    room_floor = floor_assignments[orig_j]
                    if room_floor in stairwell_connections[orig_i]:
                        can_travel = True
                elif not i_is_stairwell and j_is_stairwell:
                    room_floor = floor_assignments[orig_i]
                    if room_floor in stairwell_connections[orig_j]:
                        can_travel = True
                
                if can_travel:
                    dist_matrix[i, j] = int(base_dist_matrix[i, j] * 1000)
                else:
                    dist_matrix[i, j] = LARGE_PENALTY

    # Handle end points
    if return_to_start:
        end_points = start_points
    elif end_points is None:
        end_points = start_points
    elif isinstance(end_points, int) and end_points == -1:
        end_points = [-1] * num_vehicles

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, start_points, end_points)
    routing = pywrapcp.RoutingModel(manager)

    # --- 1. Distance Dimension (Cost of Travel) ---
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    routing.AddDimension(
        transit_callback_index,
        0, 10**10, True, 'Distance'
    )
    distance_dimension = routing.GetDimensionOrDie('Distance')
    
    # REDUCED distance span cost (we care less about distance balance now)
    distance_dimension.SetGlobalSpanCostCoefficient(10) 

    # --- 2. NEW: Counter Dimension (Balance Number of Visits) ---
    # We want to count how many "Task Nodes" (Rooms) each agent visits.
    # We do NOT want to count stairwells or the depot as "tasks".
    
    def count_evaluator(from_index):
        """Returns 1 if the node is a room, 0 if it is a stairwell or depot"""
        from_node = manager.IndexToNode(from_index)
        original_idx = get_original_idx(from_node)
        
        # Do not count stairwells or start points as "work"
        if original_idx in stairwell_set or original_idx in set(start_points):
            return 0
        return 1

    count_callback_index = routing.RegisterUnaryTransitCallback(count_evaluator)
    
    routing.AddDimension(
        count_callback_index,
        0,      # null capacity slack
        1000,   # max capacity (max nodes per agent)
        True,   # start cumul to zero
        'Counter'
    )
    
    counter_dimension = routing.GetDimensionOrDie('Counter')
    
    # A: FORCE BALANCE (The "Socialist" Clause)
    # Penalize the difference between the busiest and laziest agent.
    # We use a massive coefficient here to make imbalance strictly forbidden.
    counter_dimension.SetGlobalSpanCostCoefficient(1000000) 

    # B: FORCE PARTICIPATION (The "Mandatory Attendance" Clause)
    # This is the specific fix for your "only 4 agents" problem.
    # We tell the solver: "Every vehicle must have a Counter value of at least 1 at the end."
    penalty_for_lazy_agent = 10000000  # 10 million penalty
    min_rooms_per_agent = 1
    
    for vehicle_id in range(num_vehicles):
        # Apply this constraint to the END node of every vehicle
        index = routing.End(vehicle_id)
        counter_dimension.SetCumulVarSoftLowerBound(
            index, 
            min_rooms_per_agent, 
            penalty_for_lazy_agent
        )

    # C: DE-PRIORITIZE DISTANCE
    # If we care about balance, we must stop caring about distance.
    # Set the distance span cost to 0 so the solver doesn't try to save fuel.
    distance_dimension.SetGlobalSpanCostCoefficient(0)

    # Optional: Set Soft Lower Bound to force usage
    # This says: "You pay a penalty if you visit fewer than 1 node"
    # for vehicle_id in range(num_vehicles):
    #     counter_dimension.SetCumulVarSoftLowerBound(routing.End(vehicle_id), 2, 100000)

    # Mandatory nodes setup
    mandatory_nodes = []
    for node in range(original_n):
        if node not in stairwell_set and node not in set(start_points):
            mandatory_nodes.append(node)
    
    for node in mandatory_nodes:
        index = manager.NodeToIndex(node)
        routing.AddDisjunction([index], 10**15)
    
    # Virtual stairwells logic (same as before)
    for agent_id in range(num_vehicles):
        for stairwell_idx in stairwell_indices:
            for copy_num in range(copies_per_stairwell):
                virtual_idx = virtual_stairwell_map[(agent_id, stairwell_idx, copy_num)]
                index = manager.NodeToIndex(virtual_idx)
                routing.SetAllowedVehiclesForIndex([agent_id], index)
                routing.AddDisjunction([index], 0)

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC 
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = time_limit
    
    solution = routing.SolveWithParameters(search_parameters)

    # Extract solution (unchanged)
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
                original_node = get_original_idx(node)
                route.append(original_node)
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            
            node = manager.IndexToNode(index)
            original_node = get_original_idx(node)
            route.append(original_node)
            
            routes.append(route)
            route_lengths.append(route_distance / 1000.0)
            max_route_distance = max(max_route_distance, route_distance / 1000.0)
            if len(route) > 2: # Only mark used if they actually went somewhere
                used_starts.add(route[0])
        
        all_starts = set(start_points)
        unused_starts = all_starts - used_starts
        
        return {
            'max_route_length': max_route_distance,
            'routes': routes,
            'route_lengths': route_lengths,
            'points': points,
            'unused_starts': unused_starts,
            'floor_assignments': floor_assignments,
            'stairwell_indices': stairwell_indices
        }
    return None

# --- PLOTTING CODE (Unchanged from your provided snippet) ---
def plot_solution(solution):
    points = solution['points']
    routes = solution['routes']
    stairwell_indices = solution.get('stairwell_indices', [])
    
    plt.figure(figsize=(14, 10))
    points_array = np.array(points)
    plt.scatter(points_array[:, 0], points_array[:, 1], c='gray', s=100, alpha=0.5)
    
    if stairwell_indices:
        stairwell_points = points_array[stairwell_indices]
        plt.scatter(stairwell_points[:, 0], stairwell_points[:, 1], 
                    c='blue', s=300, marker='^', edgecolor='black', 
                    linewidth=2, label='Stairwells', zorder=15)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
    for i, (route, color) in enumerate(zip(routes, colors)):
        if not route or len(route) <= 2: 
            continue
        route_points = points_array[route]
        # Calculate actual visits (excluding start/end/stairwells)
        visits = sum(1 for x in route[1:-1] if x not in stairwell_indices)
        plt.plot(route_points[:, 0], route_points[:, 1], 'o-', color=color, 
                 linewidth=2, markersize=8, 
                 label=f'Agent {i+1}: Dist={solution["route_lengths"][i]:.1f}, Visits={visits}')
        
        start = route[0]
        end = route[-1]
        plt.scatter(points_array[start, 0], points_array[start, 1], 
                    c='green', s=200, marker='*', edgecolor='black', zorder=10)
        if start != end:
            plt.scatter(points_array[end, 0], points_array[end, 1], 
                        c='red', s=200, marker='s', edgecolor='black', zorder=10)
    
    plt.legend(loc='upper right')
    plt.title(f'MTSP Balanced Workload Solution')
    plt.grid(True)
    plt.axis('equal')
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
    # Example usage with floors separated by stairwells
    points2 = [
        [0, 0], [72, 0], [36, 0], [6, 3], [18, 3], [30, 3], 
        [42, 3], [54, 3], [66, 3], 
        [78, 3], [90, 3], [102, 3],
        [6, -3], [18, -3], [30, -3],
        [42, -3], [54, -3], [66, -3],
        [78, -3], [90, -3], [102, -3]
    ]

    points3 = [
    # ---------- FLOOR 1 (x < 40) ----------
    [0, 0],      [40, 0],    # 0: entrance / first floor reference
    [5, 4],      # 1: room 1 (upper left cluster)
    [10, 6],     # 2: room 2
    [12, 2],     # 3: room 3
    [18, -3],    # 4: room 4 (lower)
    [22, 1],     # 5: room 5
    [28, 5],     # 6: room 6
    [32, -4],    # 7: room 7

    # ---------- STAIRWELL (SEPARATOR) ----------
     # 8: stairwell connecting floor 1 and floor 2

    # ---------- FLOOR 2 (x > 40) ----------
    [50, 0],     # 9: central corridor point
    [55, 3],     # 10: room A
    [60, 3],     # 11: room B
    [65, 3],     # 12: room C
    [55, -3],    # 13: room D
    [60, -1],    # 14: room E
    [70, -4],    # 15: room F
    [78, 2],     # 16: room G
    [85, 5],     # 17: room H
    [90, 1]      # 18: room I
]

    stairwell_indices = [1, 2]
    
    # Increase to 5 agents to test utilization
    start_points = [0, 0, 0, 0, 0] 
    end_points = [0, 0, 0, 0, 0]

    print(f"Solving with {len(start_points)} agents...")
    
    solution = solve_flexible_mtsp(
        points=points2,
        start_points=start_points,
        end_points=end_points,
        stairwell_indices=stairwell_indices,
        time_limit=30,
        return_to_start=True
    )

    if solution:
        print("\nSolution found!")
        print("-" * 50)
        for i, route in enumerate(solution['routes']):
            print(f"Agent {i+1} path: {route}")
        print("-" * 50)
        
        # Export the solution to JSON
        export_solution(solution)
        
        # Show the plot
        plot_solution(solution)