# Build a per-face metric G2 = I + (∇f)(∇f)^T from a bump f based on distance to an obstacle.
# Reuses helpers from your metric-ready heat method module:
#   Heat.load_mesh, Heat.build_primitives, Heat.assemble_operators,
#   Heat.heat_distance_auto_metric, Heat.distance_at_point, Heat.nearest_vertex, etc.

from pathlib import Path
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import Heat_method_distance as Heat 


# --------------------------
# 1) f construction
# --------------------------
def f_vertex_values_from_phi(phi, M=10.0, D=0.2):
    """
    Given per-vertex distances phi (from the obstacle), return per-vertex values
    s_i = M / (1 + (phi_i / D)^2).  These are the vertex samples of f that will
    be linearly interpolated on faces.
    """
    phi = np.asarray(phi, float)
    return M / (1.0 + (phi / float(D))**2)


def f_at_point(mesh, V, F, s_vertex, x, prox=None):
    """
    Evaluate f(x) by barycentric interpolation of the per-vertex values s_vertex.
    """
    if prox is None:
        prox = trimesh.proximity.ProximityQuery(mesh)
    result = prox.on_surface([np.asarray(x, float)])
    # Handle trimesh returning (pts, dists, faces) or (pts, faces)
    if len(result) == 3:
        closest_pts, _, face_idx = result
    else:
        closest_pts, face_idx = result
    q = closest_pts[0]
    f = int(face_idx[0])
    vids = F[f]
    a, b, c = V[vids]
    lam = Heat.barycentric_coords(q, a, b, c)
    return float(lam @ s_vertex[vids]), {"q": q, "face": f, "vids": vids, "lam": lam}


# --------------------------
# 2) Metric assembly: G2 = I + (∇f)(∇f)^T in face frame (X,Y)
# --------------------------
def build_metric_G2_from_f(V, F, grads, X, Y, s_vertex):
    """
    Given per-vertex samples s = f|_vertices and per-face gradient bases,
    compute per-face metric G2[f] = I + g g^T where g = (∇f) expressed in (X,Y).

    grads: (g0,g1,g2) with each gk[f] a 3D gradient of barycentric hat function on face f
    X,Y  : per-face orthonormal frame (each (m,3))
    s_vertex: (n,) per-vertex f values (the M/(1+(phi/D)^2) samples)
    Returns: G2 (m,2,2) SPD per-face matrices usable by Heat.assemble_operators
    """
    g0, g1, g2 = grads
    m = len(F)
    G2 = np.zeros((m, 2, 2), dtype=float)

    for f in range(m):
        i, j, k = F[f]
        # ∇f on face f in 3D: linear comb of basis gradients with vertex samples
        gradf3 = s_vertex[i] * g0[f] + s_vertex[j] * g1[f] + s_vertex[k] * g2[f]
        # project to 2D frame (X[f], Y[f])
        gx = float(np.dot(gradf3, X[f]))
        gy = float(np.dot(gradf3, Y[f]))
        g = np.array([gx, gy])
        # G = I + g g^T
        G2[f] = np.eye(2) + np.outer(g, g)

    return G2


# --------------------------
# 3) Example pipeline (usage)
# --------------------------
def main():
    # -----------------------
    # Paths & imports
    # -----------------------
    from trimesh.path import entities as path_entities
    from trimesh.path import Path3D

    # Fallback helpers if your Heat module doesn't expose these
    def path3d_from_polyline(points, color=(0, 0, 0, 255)):
        pts = np.asarray(points, float)
        ents = [path_entities.Line(np.array([i, i + 1], dtype=int))
                for i in range(len(pts) - 1)]
        path = Path3D(entities=ents, vertices=pts, process=False)
        path.colors = np.tile(np.array(color, dtype=np.uint8), (len(ents), 1))
        return path

    def small_sphere(center, radius=0.02, rgba=(255, 0, 0, 255)):
        s = trimesh.creation.uv_sphere(radius=radius)
        s.apply_translation(center)
        s.visual.vertex_colors = np.tile(np.array(rgba, dtype=np.uint8), (len(s.vertices), 1))
        return s

    # -----------------------
    # 0) Load mesh & primitives
    # -----------------------
    obj_path = Path("C:/Users/jdkg7/Documents/Python/Metric Modification/torus.obj")
    mesh, V, F = Heat.load_mesh(obj_path)
    A, grads, X, Y, h = Heat.build_primitives(mesh, V, F)

    # Euclidean operators (G2=None means identity metric per face)
    K_euc, M_euc, Ginv2_euc, detsqrt_euc = Heat.assemble_operators(
        V, F, A, grads, X, Y, G2=None
    )

    # -----------------------
    # 1) Set guesses & project to surface
    # -----------------------
    src_rough = np.array([0.8, -0.2, -0.])
    tgt_rough = np.array([-1.1, -0.5, -0.3])
    obs_rough = np.array([0.1, -0.9, -0.1])

    prox = trimesh.proximity.ProximityQuery(mesh)

    # project to mesh surface
    p_proj   = (prox.on_surface([obs_rough ])[0])[0]
    src_proj = (prox.on_surface([src_rough])[0])[0]
    tgt_proj = (prox.on_surface([tgt_rough])[0])[0]

    # Choose vertices for obstacle (for φ_obstacle) and source (for distances)
    v_obs = Heat.nearest_vertex(V, p_proj)     # obstacle's nearest vertex
    v_src = Heat.nearest_vertex(V, src_proj)   # source's nearest vertex

    # -----------------------
    # 2) STEP 1: Euclidean distance φ_obstacle = d(·, obstacle)
    #    (used to build the bump f)
    # -----------------------
    phi_obstacle, t_euc_obs, _ = Heat.heat_distance_auto_metric(
        V, F, A, grads, X, Y, K_euc, M_euc, Ginv2_euc, detsqrt_euc, v_obs, h
    )

    # Build per-vertex samples of f: s_i = M/(1+(phi_i/D)^2)
    # Double torus: M = 4, D = 0.6, k = 2
    # Single tours: M = 0.8, D = 0.1, k = 2  (Same homotopy class)
    # Single torus: M = 2, D = 0.4, k = 2  (same leg)
    # Single torus: M = 4, D = 0.7, k = 2 (opposite leg)
    M_param = 0.8
    D_param = 0.1
    s_vertex = f_vertex_values_from_phi(phi_obstacle, M=M_param, D=D_param)

    # -----------------------
    # 3) STEP 2: Build metric G2 = I + (∇f)(∇f)^T per face (in frame (X,Y))
    # -----------------------
    G2 = build_metric_G2_from_f(V, F, grads, X, Y, s_vertex)

    # Assemble operators for modified metric
    K_mod, M_mod, Ginv2_mod, detsqrt_mod = Heat.assemble_operators(
        V, F, A, grads, X, Y, G2=G2
    )

    # -----------------------
    # 4) STEP 3: Distances source→target for:
    #            (a) Euclidean metric
    #            (b) Modified metric
    # -----------------------
    # (a) Euclidean distance field from source
    dist_euc_src, t_euc_src, _ = Heat.heat_distance_auto_metric(
        V, F, A, grads, X, Y, K_euc, M_euc, Ginv2_euc, detsqrt_euc, v_src, h
    )
    d_euc_tgt, _ = Heat.distance_at_point(mesh, V, F, dist_euc_src, tgt_proj, prox=prox)

    # (b) Modified metric distance field from source
    dist_mod_src, t_mod_src, _ = Heat.heat_distance_auto_metric(
        V, F, A, grads, X, Y, K_mod, M_mod, Ginv2_mod, detsqrt_mod, v_src, h
    )
    d_mod_tgt, _ = Heat.distance_at_point(mesh, V, F, dist_mod_src, tgt_proj, prox=prox)

    # -----------------------
    # 5) Print summary
    # -----------------------
    print(f"Loaded mesh: {len(V)} vertices, {len(F)} faces; watertight={mesh.is_watertight}")
    print(f"Obstacle (projected): {p_proj}")
    print(f"Source   (projected): {src_proj}  [v={v_src}]")
    print(f"Target   (projected): {tgt_proj}")
    print("\n--- Distances source→target ---")
    print(f"Euclidean metric   : {d_euc_tgt:.6f}  (t={t_euc_src:.2e})")
    print(f"Modified metric G  : {d_mod_tgt:.6f}  (t={t_mod_src:.2e})")

    # -----------------------
    # 6) Trace geodesic under modified metric (steepest descent of dist_mod_src)
    # -----------------------
    v_src = int(Heat.nearest_vertex(V, src_proj))

    path_pts, path_len, hit_src = Heat.trace_geodesic_polyline_metric(
        mesh, V, F,
        phi=dist_euc_src,
        grads=grads,
        X=X, Y=Y,
        Ginv2=Ginv2_euc,          # <-- required
        start_point=tgt_proj,
        src_idx=v_src,            # <-- integer vertex id
        max_steps=10000,
        step_tol=1e-10,
        grad_tol=1e-14,
        eps=1e-14,
        phi_tol_factor=5e-3,
        verbose=True
    )
    print(f"Geodesic traced: length ≈ {path_len:.4f}, hit_source={hit_src}")

    path_pts_mod, path_len_mod, hit_src_mod = Heat.trace_geodesic_polyline_metric(
        mesh, V, F,
        phi=dist_mod_src,
        grads=grads,
        X=X, Y=Y,
        Ginv2=Ginv2_mod,          # <-- required
        start_point=tgt_proj,
        src_idx=v_src,            # <-- integer vertex id
        max_steps=10000,
        step_tol=1e-10,
        grad_tol=1e-14,
        eps=1e-14,
        phi_tol_factor=5e-3,
        verbose=True
    )
    print(f"Modified Geodesic traced: length ≈ {path_len_mod:.4f}, hit_source={hit_src_mod}")

    # -----------------------
    # 7) Visualization: color by modified distance; draw source/target/obstacle and path
    # -----------------------
    # Distance colormap (coolwarm: red=near, blue=far), invert so red near source
    dn = 1.0 - (dist_mod_src - dist_mod_src.min()) / (dist_mod_src.max() - dist_mod_src.min() + 1e-16)
    colors = (plt.colormaps.get_cmap('coolwarm')(dn)[:, :3] * 255).astype(np.uint8)
    mesh.visual.vertex_colors = colors

    # Spheres
    obs_ball = small_sphere(p_proj,   radius=0.02, rgba=(255, 0,   0, 255))  # red: obstacle
    col_zone = small_sphere(p_proj,   radius=D_param*0.9, rgba=(255, 255, 255, 100))  # grey: collision zone
    src_ball = small_sphere(src_proj, radius=0.02, rgba=(0,   255, 0, 255))  # green: source
    tgt_ball = small_sphere(tgt_proj, radius=0.02, rgba=(0,   0, 255, 255))  # blue: target

    # Polyline
    geodesic_obj = path3d_from_polyline(path_pts, color=(0, 0, 0, 135))          # grey
    mod_geodesic_obj = path3d_from_polyline(path_pts_mod, color=(0, 0, 0, 255))          # black

    # Show scene
    scene = trimesh.Scene([mesh, geodesic_obj, mod_geodesic_obj, obs_ball, col_zone, src_ball, tgt_ball])
    scene.show()

     # -----------------------
    # 8) Matplotlib figure (paper-ready)
    # -----------------------
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    # Create figure
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 0.6])  # nice proportions for torus

    # Prepare face colors from the colormap already assigned
    face_colors = colors[F].mean(axis=1) / 255.0

    # Add surface mesh (semi-transparent for depth)
    mesh_tris = V[F]
    surf = Poly3DCollection(mesh_tris, facecolors=face_colors, linewidths=0.0, alpha=0.95)
    ax.add_collection3d(surf)

    # Add paths
    ax.plot(path_pts[:, 0], path_pts[:, 1], path_pts[:, 2],
            color='gray', lw=1.2, label='Euclidean geodesic')
    ax.plot(path_pts_mod[:, 0], path_pts_mod[:, 1], path_pts_mod[:, 2],
            color='black', lw=1.8, label='Modified geodesic')

    # Add obstacle zone (semi-transparent sphere)
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = D_param * 0.9 * np.cos(u) * np.sin(v) + p_proj[0]
    y = D_param * 0.9 * np.sin(u) * np.sin(v) + p_proj[1]
    z = D_param * 0.9 * np.cos(v) + p_proj[2]
    ax.plot_surface(x, y, z, color='lightgray', alpha=0.3, linewidth=0)

    # Source, target, obstacle points
    ax.scatter(*src_proj, color='green', s=35, label='Source', depthshade=False)
    ax.scatter(*tgt_proj, color='blue', s=35, label='Target', depthshade=False)
    ax.scatter(*p_proj, color='red', s=35, label='Obstacle', depthshade=False)

    # Camera and lighting style
    ax.view_init(elev=25, azim=50)
    ax.axis('off')
    ax.legend(frameon=False, loc='upper left')

    # Optional: equalize axis scale for a true torus shape
    def set_axes_equal(ax):
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        spans = limits[:, 1] - limits[:, 0]
        centers = np.mean(limits, axis=1)
        max_span = max(spans)
        new_limits = np.array([centers - max_span / 2, centers + max_span / 2]).T
        ax.set_xlim3d(new_limits[0])
        ax.set_ylim3d(new_limits[1])
        ax.set_zlim3d(new_limits[2])
    set_axes_equal(ax)

    # Save high-resolution figure
    fig.tight_layout()
    fig.savefig("torus_geodesics.png", dpi=600, bbox_inches='tight', transparent=True)
    fig.savefig("torus_geodesics.pdf", bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == "__main__":
    main()