# Metric-modified-heat-method
The code runs the heat method of Crane et. al. to approximate Riemannian distances on a triangulated mesh. We apply the method with a modified Riemannian metric chosen so that distances inflate near a prescribed obstacle. Geodesics of this metric avoid the obstacle, and are constructed by applying gradient descent to the Riemannian distance.

The heat method can be broken into three phases:
1) Solve the heat equation with dirac-delta initial condition (centered at some chosen source point) to approximate heat kernel at that point.
2) Normalize the heat kernel from the previous step.
3) Solve the Poisson equation, where the source function is the normalized heat kernel.
The end result of this method is an approximation to the Riemannian distance function from the source point. See the paper "Geodesics in Heat" by Crane, Weischedel, and Wardetzky for more details on the heat method.

Heat_method_distance.py contains most of the tools used to run the ordinary heat method. 
The code accepts a triangulated mesh in the form of a .obj file. I have included three such files in the repository: a sphere, a torus, and a double torus. Such meshes can be easily created in Blender, but be sure that the mesh is watertight to avoid strange gradient behaviors. In the code, a finite element method (FEM) is applied with respect to the P1 nodal basis of a triangulated mesh to convert the heat equation and Poisson equation into sparse linear systems. By default, it is assumed that the Riemannian metric on your triangulated mesh is the metric induced by its embedding into Euclidean space, but it is possible to input non-trivial face-wise linear metrics. Geodesics between a given source and target point are then drawn by applying gradient descent to the approximated Riemannian distance function (note: exact locations of source and target points are not needed, as the code will snap to nearest point on mesh by default). If the gradient of the Riemannian distance becomes sufficiently small at some step, Dijkstra's algorithm will automatically be applied instead to avoid the geodesics from stalling out. If run by itself, the code will produce a 3D representation of the inputted mesh, with the source and target points marked, and the constructed minimal-length geodesic (hopefully) connecting them.

Heat_method_obstacle_avoidance.py imports Heat_method_distance.py to run the heat method and draw geodesics. Additionally, a function is inputted to distort the geometry of the manifold. The process is as follows:
1) A function f is chosen, and calculated on the vertex set.
2) The unique face-wise linear function F agreeing with f on the vertex set is constructed.
3) the modified Riemannian metric G = g + dF⊗dF is constructed on the mesh, where g denotes the ordinary metric induced by inclusion.
4) The heat method is run with the metric G to construct a modified geodesic.
In practice, the function f will depend on the Riemannian distance of g, so that a first pass of the heat method is run where the chosen source point is an obstacle that we wish to avoid, which we then use to construct f.

