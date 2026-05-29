import rerun as rr
import numpy as np

rr.init("my_rerun_graph_example", spawn=True)
# Define node positions (e.g., in 3D space)
positions = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
])

# Define edges (connections between nodes)
# Each pair of indices represents an edge
edges = np.array([
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
])
# Log the nodes as 3D points
rr.log("graph/nodes", rr.Points3D(positions, radii=0.05))

# Log the edges as 3D line strips
# For a graph, you often want to connect specific points,
# so you'd structure your line strips accordingly.
# Here, we'll create a list of line strips based on the edges.
line_strips = []
for edge in edges:
    line_strips.append(positions[edge])
rr.log("graph/edges", rr.LineStrips3D(line_strips, stroke_width=0.02))

