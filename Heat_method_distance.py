# Heat method on a triangulated surface with a user metric G (defaults to identity)
# Uses: pathlib, os, numpy, scipy, trimesh, matplotlib
from pathlib import Path
import os
import numpy as np
import trimesh
from trimesh.path import entities as path_entities
from trimesh.path import Path3D
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import heapq


# --------------------------
# Load mesh
# --------------------------
def load_mesh(obj_path: Path):
    mesh = trimesh.load(obj_path, process=False)
    V = mesh.vertices.astype(np.float64)
    F = mesh.faces.astype(np.int64)
    return mesh, V, F


# --------------------------
# Plot helpers
# --------------------------
def path3d_from_polyline(points, color=(0, 0, 0, 255)):
    """
    Build a Trimesh Path3D (polyline) from an (N,3) array of points.
    """
    points = np.asarray(points, float)
    if len(points) < 2:
        raise ValueError("Need at least 2 points to build a path.")
    ents = [path_entities.Line(np.array([i, i+1], dtype=int))
            for i in range(len(points)-1)]
    path = Path3D(entities=ents, vertices=points, process=False)
    # Color the whole path (one color per entity)
    path.colors = np.tile(np.array(color, dtype=np.uint8), (len(ents), 1))
    return path

def small_sphere(center, radius=0.02, rgba=(255, 0, 0, 255)):
    s = trimesh.creation.uv_sphere(radius=radius)
    s.apply_translation(center)
    s.visual.vertex_colors = np.tile(np.array(rgba, dtype=np.uint8), (len(s.vertices), 1))
    return s


# --------------------------
# Geometry primitives (metric-agnostic)
# --------------------------
def build_primitives(mesh, V, F):
    """
    Returns:
      A      : (m,) per-face Euclidean area
      grads  : (g0,g1,g2) with shape (m,3), per-face ∇φ_i in 3D
      X,Y    : (m,3) per-face orthonormal frame in the tangent plane
      h      : mean edge length (for t = c h^2)
    """
    vi, vj, vk = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

    # Face normal and area
    eij = vj - vi
    eik = vk - vi
    n = np.cross(eij, eik)
    dblA = np.linalg.norm(n, axis=1)  # 2*area
    A = 0.5 * dblA
    nhat = n / np.maximum(dblA, 1e-16)[:, None]

    # ∇φ_i on each face: ∇φ_i = (n̂ × e_opposite) / ||n||
    e0 = vk - vj  # opposite vertex 0
    e1 = vi - vk  # opposite vertex 1
    e2 = vj - vi  # opposite vertex 2
    g0 = np.cross(nhat, e0) / np.maximum(dblA, 1e-16)[:, None]
    g1 = np.cross(nhat, e1) / np.maximum(dblA, 1e-16)[:, None]
    g2 = np.cross(nhat, e2) / np.maximum(dblA, 1e-16)[:, None]

    # Per-face orthonormal frame: X along (vj-vi), Y = n × X
    X = eij / np.maximum(np.linalg.norm(eij, axis=1, keepdims=True), 1e-15)
    n_hat = nhat
    Y = np.cross(n_hat, X)
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-15)

    # Mean edge length
    h = float(mesh.edges_unique_length.mean())

    return A, (g0, g1, g2), X, Y, h


# --------------------------
# Metric assembly (always used; identity if G2=None)
# --------------------------
def assemble_operators(V, F, A, grads, X, Y, G2=None):
    """
    Assemble stiffness K and lumped mass M for a per-face metric G2 (m,2,2) in (X,Y) frame.
      K_ij^F = (∇φ_i)^T G_F^{-1} ∇φ_j * sqrt(det G_F) * A_F
      M_ii^F = sqrt(det G_F) * A_F / 3
    If G2 is None, uses identity (Euclidean weak form).
    Returns: K (csr), M (diag), Ginv2 (m,2,2), det_sqrt (m,)
    """
    g0, g1, g2 = grads
    m = len(F); n = len(V)

    if G2 is None:
        # Identity metric per face
        G2 = np.repeat(np.eye(2)[None, :, :], m, axis=0)

    Ginv2 = np.linalg.inv(G2)                  # (m,2,2)
    det_sqrt = np.sqrt(np.linalg.det(G2))      # (m,)

    rows, cols, vals = [], [], []
    Mi = np.zeros(n, dtype=float)

    def proj2(q3, Xf, Yf):
        return np.array([np.dot(q3, Xf), np.dot(q3, Yf)], dtype=float)

    def add(i, j, w):
        rows.append(i); cols.append(j); vals.append(w)

    for f in range(m):
        i, j, k = F[f]
        aF = A[f] * det_sqrt[f]

        # project per-face gradients to 2D frame
        q0 = proj2(g0[f], X[f], Y[f])
        q1 = proj2(g1[f], X[f], Y[f])
        q2_ = proj2(g2[f], X[f], Y[f])
        Q = (q0, q1, q2_); idx = (i, j, k)

        # local 3×3 block for stiffness
        for u in range(3):
            for v in range(3):
                kij = aF * (Q[u] @ (Ginv2[f] @ Q[v]))
                add(idx[u], idx[v], kij)

        # lumped mass
        Mi[i] += aF/3.0; Mi[j] += aF/3.0; Mi[k] += aF/3.0

    K = csr_matrix((vals, (rows, cols)), shape=(n, n))
    M = diags(Mi)
    return K, M, Ginv2, det_sqrt


# --------------------------
# Metric unit field + RHS (no conformal factor)
# --------------------------
def metric_vector_field_and_rhs(V, F, A, grads, X, Y, G2, Ginv2, detsqrt, u):
    """
    Given scalar u on vertices and metric (G2,Ginv2,detsqrt) per face in the (X,Y) frame,
    compute:
      - X3: 3D unit descent field (one vector per face)
      - b : divergence RHS for K φ = b (assembled with metric area)
    """
    g0, g1, g2b = grads
    i, j, k = F[:, 0], F[:, 1], F[:, 2]

    # Per-face grad u in 3D
    grad3 = u[i][:, None] * g0 + u[j][:, None] * g1 + u[k][:, None] * g2b  # (m,3)

    # Project grad to local 2D
    ux = np.einsum('ij,ij->i', grad3, X)
    uy = np.einsum('ij,ij->i', grad3, Y)
    grad2 = np.stack([ux, uy], axis=1)  # (m,2)

    # Metric gradient ∇_G u = G^{-1} grad2
    w2 = np.einsum('mij,mj->mi', Ginv2, grad2)       # (m,2)
    # Metric norm ||∇u||_G = sqrt( grad2^T G^{-1} grad2 )
    nG = np.sqrt(np.einsum('mi,mij,mj->m', grad2, Ginv2, grad2)) + 1e-15

    # Unit field in 2D: -w2 / ||∇u||_G
    d2 = - (w2 / nG[:, None])

    # Lift to 3D: X3 = d2.x * X + d2.y * Y
    X3 = d2[:, 0][:, None] * X + d2[:, 1][:, None] * Y  # (m,3)

    # Divergence RHS (weak form): b_i += ∑_f (A_f * sqrt(det G_f)) <∇φ_i, X3_f>
    Af = A * detsqrt  # metric area per face
    dot0 = np.einsum('ij,ij->i', g0, X3)
    dot1 = np.einsum('ij,ij->i', g1, X3)
    dot2 = np.einsum('ij,ij->i', g2b, X3)

    b = np.zeros(len(V), dtype=np.float64)
    np.add.at(b, i, Af * dot0)
    np.add.at(b, j, Af * dot1)
    np.add.at(b, k, Af * dot2)

    return X3, b


# --------------------------
# Heat method (metric)
# --------------------------
def heat_method_once_metric(V, F, A, grads, X, Y, K, M, Ginv2, det_sqrt, src, t):
    """
    One pass of the heat method under a user metric:
      (M + tK) u = M u0
      X_f = - G^{-1} ∇u / ||∇u||_G
      K φ = div(X) with φ[src]=0
    Returns: distance field φ normalized so φ[src]=0.
    """
    n = len(V)

    # Unit-mass delta (metric mass)
    u0 = np.zeros(n)
    u0[src] = 1.0 / M.diagonal()[src]

    # Heat step
    u = spsolve(M + t * K, M @ u0)

    # Unit field & RHS under the metric
    # (reconstruct G2 only for shape, not used here)
    m = len(F)
    G2_dummy = np.repeat(np.eye(2)[None, :, :], m, axis=0)
    _, b = metric_vector_field_and_rhs(V, F, A, grads, X, Y, G2_dummy, Ginv2, det_sqrt, u)

    # Anchor φ[src]=0
    Kanch = K.tolil()
    Kanch[src, :] = 0.0; Kanch[:, src] = 0.0; Kanch[src, src] = 1.0
    phi = spsolve(Kanch.tocsr(), np.where(np.arange(n) == src, 0.0, b))
    return phi - phi[src]


def heat_distance_auto_metric(V, F, A, grads, X, Y, K, M, Ginv2, det_sqrt, src, h,
                              candidates=(1, 2, 5, 10, 20, 50)):
    """
    Try t = c h^2; choose by maximizing dynamic range of φ.
    Returns: dist_best, t_best, records
    """
    records = []
    for c in candidates:
        t_val = float(c) * (h * h)
        phi = heat_method_once_metric(V, F, A, grads, X, Y, K, M, Ginv2, det_sqrt, src, t_val)
        dyn = float(phi.max() - phi.min())
        records.append((dyn, c, t_val, phi))
    records.sort(reverse=True)  # prefer larger dynamic range
    _, c_best, t_best, dist_best = records[0]
    return dist_best, t_best, records


# --------------------------
# Interpolation helpers
# --------------------------
def nearest_vertex(V, point3):
    return int(np.argmin(np.linalg.norm(V - np.asarray(point3), axis=1)))

def barycentric_coords(p, a, b, c, eps=1e-16):
    v0 = b - a; v1 = c - a; v2 = p - a
    d00 = np.dot(v0, v0); d01 = np.dot(v0, v1); d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0); d21 = np.dot(v2, v1)
    denom = d00*d11 - d01*d01
    if denom <= eps:
        dists = np.array([np.linalg.norm(p - a), np.linalg.norm(p - b), np.linalg.norm(p - c)])
        lam = np.zeros(3); lam[int(np.argmin(dists))] = 1.0
        return lam
    v = (d11*d20 - d01*d21)/denom
    w = (d00*d21 - d01*d20)/denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=float)

def distance_at_point(mesh, V, F, phi, p, prox=None):
    if prox is None:
        prox = trimesh.proximity.ProximityQuery(mesh)
    result = prox.on_surface([np.asarray(p, float)])
    if len(result) == 3:
        closest_pts, _, face_idx = result
    else:
        closest_pts, face_idx = result
    q = closest_pts[0]
    f = int(face_idx[0])
    vids = F[f]
    a, b, c = V[vids]
    lam = barycentric_coords(q, a, b, c)
    return float(lam @ phi[vids]), {"q": q, "face": f, "vids": vids, "lam": lam}


# --------------------------
# Face adjacency + metric-aware path tracer + helpers
# --------------------------
def build_face_adjacency(F):
    F = np.asarray(F); m = len(F)
    def key(a, b): return (a, b) if a < b else (b, a)
    E = np.stack([np.c_[F[:,1], F[:,2]], np.c_[F[:,2], F[:,0]], np.c_[F[:,0], F[:,1]]], axis=1)
    edge2faces = {}
    for f in range(m):
        for e in range(3):
            edge2faces.setdefault(key(E[f,e,0], E[f,e,1]), []).append((f, e))
    neighbors = np.full((m,3,2), -1, dtype=int)
    for faces in edge2faces.values():
        if len(faces) == 2:
            (f0,e0), (f1,e1) = faces
            neighbors[f0,e0] = (f1,e1); neighbors[f1,e1] = (f0,e0)
    return neighbors

def face_gradient_phi(phi, F, g0, g1, g2, f):
    i, j, k = F[f]
    return phi[i]*g0[f] + phi[j]*g1[f] + phi[k]*g2[f]

def intersect_ray_triangle_edges_2d(p2, v2, tri2, eps=1e-12):
    """
    Intersect the forward ray  p2 + t v2  (t>0) with the 3 edges of a triangle in 2D.
    tri2: (3,2) array with vertices [a2,b2,c2] in the same order as the face.
    Returns: (t_hit, q2, edge_index) for the smallest positive t, or (None, None, None)
    if no forward intersection exists.
    Edge indices: 0:(a,b), 1:(b,c), 2:(c,a)
    """
    hits = []
    a2, b2, c2 = tri2
    edges = [(a2, b2), (b2, c2), (c2, a2)]

    for ei, (A, B) in enumerate(edges):
        w = B - A  # edge direction
        # Solve p2 + t v2 = A + u w  ->  [v2, -w] [t,u]^T = A - p2
        M = np.array([v2, -w]).T  # 2x2
        rhs = A - p2
        det = M[0, 0]*M[1, 1] - M[0, 1]*M[1, 0]
        if abs(det) < eps:
            continue  # parallel or nearly so
        inv = (1.0/det) * np.array([[ M[1,1], -M[0,1]],
                                    [-M[1,0],  M[0,0]]])
        t, u = inv @ rhs
        # forward ray (t>0) and within the segment (0<=u<=1)
        if t > eps and u >= -eps and u <= 1.0 + eps:
            q2 = p2 + t * v2
            hits.append((t, q2, ei))

    if not hits:
        return None, None, None

    t_hit, q2, ei = min(hits, key=lambda x: x[0])
    return t_hit, q2, ei

def finish_segment_to_vertex(mesh, V, F, start_p, start_face, target_vid, X, Y,
                             max_hops=1000, eps=1e-12):
    """
    Walk the straight segment from start_p (in face 'start_face') to the vertex V[target_vid],
    crossing faces by 2D edge intersections. Returns (pts, length).
    """
    def proj2(u, Xf, Yf, origin):
        d = u - origin
        return np.array([float(np.dot(d, Xf)), float(np.dot(d, Yf))])

    # adjacency: (face f, local edge e) -> (neighbor_face, neighbor_local_edge)
    neighbors = build_face_adjacency(F)

    p = start_p.copy()
    f = int(start_face)
    tgt = V[int(target_vid)]

    pts = [p.copy()]
    total_len = 0.0

    for hop in range(max_hops):
        # if target is one of the current face vertices, connect and finish
        i, j, k = F[f]
        if target_vid in (i, j, k):
            seg = tgt - p
            L = np.linalg.norm(seg)
            if L > eps:
                pts.append(tgt.copy())
                total_len += L
            return np.vstack(pts), float(total_len)

        # build local 2D frame on current face and project p, tgt, and face vertices
        a, b, c = V[[i, j, k]]
        origin, Xf, Yf = a, X[f], Y[f]
        p2   = proj2(p,   Xf, Yf, origin)
        t2   = proj2(tgt, Xf, Yf, origin)
        tri2 = np.array([proj2(a, Xf, Yf, origin),
                         proj2(b, Xf, Yf, origin),
                         proj2(c, Xf, Yf, origin)])
        v2 = t2 - p2
        nv = np.linalg.norm(v2)
        if nv < eps:
            # already at target
            pts.append(tgt.copy())
            return np.vstack(pts), float(total_len)

        v2 /= nv  # unit direction toward target

        # intersect forward ray with triangle edges
        t_hit, q2, e_idx = intersect_ray_triangle_edges_2d(p2, v2, tri2)
        if t_hit is None:
            # fallback: just connect to target (should only happen extremely close)
            seg = tgt - p
            L = np.linalg.norm(seg)
            if L > eps:
                pts.append(tgt.copy())
                total_len += L
            return np.vstack(pts), float(total_len)

        # 3D point of intersection
        q3 = origin + q2[0] * Xf + q2[1] * Yf
        seg_len = np.linalg.norm(q3 - p)
        if seg_len > eps:
            pts.append(q3.copy())
            total_len += seg_len

        # move to neighbor across that local edge
        nb_f, _ = neighbors[f, e_idx]
        if nb_f < 0:
            # boundary (shouldn't happen on closed surfaces) — finish to target
            seg = tgt - q3
            L = np.linalg.norm(seg)
            if L > eps:
                pts.append(tgt.copy()); total_len += L
            return np.vstack(pts), float(total_len)

        p = q3
        f = int(nb_f)

    # max hops reached: just connect to target as a last resort
    seg = tgt - p
    L = np.linalg.norm(seg)
    if L > eps:
        pts.append(tgt.copy()); total_len += L
    return np.vstack(pts), float(total_len)


# -------- Metric Dijkstra (graph fallback) --------
def _edge_face_map(F):
    ef = {}
    def key(a,b): return (a,b) if a<b else (b,a)
    for f,(i,j,k) in enumerate(F):
        for a,b in ((i,j),(j,k),(k,i)):
            ef.setdefault(key(a,b), []).append(f)
    return ef

def _edge_metric_length(V, i, j, edge_faces, G2, X, Y):
    d = V[j] - V[i]
    L = np.linalg.norm(d)
    if L == 0.0:
        return 0.0
    e = d / L
    faces = edge_faces.get((min(i,j), max(i,j)), [])
    if not faces:
        return L  # boundary fallback
    acc = 0.0
    for f in faces:
        ex = float(np.dot(e, X[f])); ey = float(np.dot(e, Y[f]))
        u = np.array([ex, ey])
        acc += np.sqrt(max(1e-16, u @ (G2[f] @ u))) * L
    return acc / len(faces)

def dijkstra_metric_path(V, F, G2, X, Y, src, tgt):
    n = len(V)
    edge_faces = _edge_face_map(F)

    # adjacency with metric weights
    nbrs = [[] for _ in range(n)]
    for (a,b) in edge_faces.keys():
        w = _edge_metric_length(V, a, b, edge_faces, G2, X, Y)
        nbrs[a].append((b,w))
        nbrs[b].append((a,w))

    dist = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=int)
    dist[tgt] = 0.0
    pq = [(0.0, tgt)]
    seen = set()
    while pq:
        d,u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == src:
            break
        for v,w in nbrs[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    # reconstruct src -> tgt
    path_idx = []
    u = src
    if prev[u] < 0:
        return np.array([V[src], V[tgt]]), float(np.linalg.norm(V[tgt]-V[src]))
    while u >= 0:
        path_idx.append(u)
        if u == tgt:
            break
        u = prev[u]
    pts = V[path_idx]
    length = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    return pts, float(length)


# --------------------------
# Tracer with built-in Dijkstra fallback
# --------------------------
def trace_geodesic_polyline_metric(mesh, V, F, phi, grads, X, Y, Ginv2,
                                   start_point, src_idx,
                                   max_steps=10000, step_tol=1e-10, grad_tol=1e-14, eps=1e-14,
                                   phi_tol_factor=5e-3, verbose=False):
    """
    Face-by-face metric steepest-descent tracer.
    If it fails to reach the source, it falls back to a metric-aware Dijkstra
    path on the vertex graph using G2 = inv(Ginv2) (reconstructed internally).
    """
    g0, g1, g2_grad = grads
    neighbors = build_face_adjacency(F)

    pq = trimesh.proximity.ProximityQuery(mesh)
    result = pq.on_surface([np.asarray(start_point, float)])
    if len(result) == 2:   closest, face_idx = result
    else:                  closest, _, face_idx = result
    p = closest[0]
    f = int(face_idx[0])

    # start strictly interior
    i, j, k = F[f]
    a, b, c = V[[i, j, k]]
    lam = barycentric_coords(p, a, b, c)
    eps_bary = 1e-6
    if (lam <= eps_bary).any():
        lam = (1.0 - eps_bary) * lam + eps_bary * np.array([1/3, 1/3, 1/3], float)
        p = lam[0]*a + lam[1]*b + lam[2]*c

    pts = [p.copy()]
    total_len = 0.0
    hit_source = False

    h_local = float(mesh.edges_unique_length.mean())
    # tighter last-mile thresholds
    phi_tol = max(1e-7, 5e-5 * h_local)
    r_switch = 1.25 * h_local

    src_idx = int(src_idx)
    edge_locals = ((1, 2), (2, 0), (0, 1))

    for step in range(max_steps):
        i, j, k = F[f]

        # If very close, finish deterministically across faces to the source vertex
        dist_to_src = np.linalg.norm(p - V[src_idx])
        if (dist_to_src < r_switch):
            finish_pts, finish_len = finish_segment_to_vertex(
                mesh, V, F, start_p=p, start_face=f, target_vid=src_idx, X=X, Y=Y
            )
            if len(finish_pts) > 1:
                pts.extend(finish_pts[1:])     # append (skip duplicated p)
                total_len += finish_len
            hit_source = True
            break

        if src_idx in (i, j, k):
            seg = V[src_idx] - p
            L = np.linalg.norm(seg)
            if L > step_tol:
                pts.append(V[src_idx].copy())
                total_len += L
            hit_source = True
            break

        a, b, c = V[[i, j, k]]
        lam = barycentric_coords(p, a, b, c)
        phi_p = lam[0]*phi[i] + lam[1]*phi[j] + lam[2]*phi[k]
        # don't stop on small φ; finisher handles last meters

        # ∇φ → metric-steepest descent direction
        grad3 = phi[i]*g0[f] + phi[j]*g1[f] + phi[k]*g2_grad[f]
        gx = float(np.dot(grad3, X[f])); gy = float(np.dot(grad3, Y[f]))
        g2_local = np.array([gx, gy])
        u2 = Ginv2[f] @ g2_local
        v3 = -(u2[0]*X[f] + u2[1]*Y[f])
        nv = np.linalg.norm(v3)
        if nv < grad_tol:
            # gradient collapsed before switch radius: deterministic finish
            finish_pts, finish_len = finish_segment_to_vertex(
                mesh, V, F, start_p=p, start_face=f, target_vid=src_idx, X=X, Y=Y
            )
            if len(finish_pts) > 1:
                pts.extend(finish_pts[1:])
                total_len += finish_len
                hit_source = True
            break
        v = v3 / nv

        # barycentric stepping: s_l = <∇λ_l, v>
        s0 = float(np.dot(g0[f],      v))
        s1 = float(np.dot(g1[f],      v))
        s2 = float(np.dot(g2_grad[f], v))

        t_candidates = []
        for l, lam_l, s_l in ((0, lam[0], s0), (1, lam[1], s1), (2, lam[2], s2)):
            if s_l < -1e-16:
                t_l = -lam_l / s_l
                if t_l > step_tol:
                    t_candidates.append((t_l, l))

        if not t_candidates:
            # fallback tiny step; if still stuck, finish to vertex
            tiny = 1e-9 * h_local
            p_try = p + tiny * v
            result = pq.on_surface([p_try])
            if len(result) == 2:   closest, face_idx = result
            else:                  closest, _, face_idx = result
            p2 = closest[0]
            f2 = int(face_idx[0])
            if np.linalg.norm(p2 - p) > step_tol:
                pts.append(p2.copy()); total_len += np.linalg.norm(p2 - p)
                p, f = p2, f2
                if verbose: print(f"[trace] tiny fallback at {step}")
                continue
            # still stuck → deterministic finish
            finish_pts, finish_len = finish_segment_to_vertex(
                mesh, V, F, start_p=p, start_face=f, target_vid=src_idx, X=X, Y=Y
            )
            if len(finish_pts) > 1:
                pts.extend(finish_pts[1:]); total_len += finish_len
                hit_source = True
            break

        # earliest edge crossing
        t_min, l_edge = min(t_candidates, key=lambda x: x[0])

        # if the crossing edge already contains the source, snap
        sA_old, sB_old = edge_locals[l_edge]
        gA = F[f, sA_old]; gB = F[f, sB_old]
        if src_idx == gA or src_idx == gB:
            q = V[src_idx]
            seg_len = np.linalg.norm(q - p)
            if seg_len > step_tol:
                pts.append(q.copy()); total_len += seg_len
            hit_source = True
            break

        # normal march
        q = p + t_min * v
        seg_len = np.linalg.norm(q - p)
        if seg_len < step_tol:
            # too tiny → finish
            finish_pts, finish_len = finish_segment_to_vertex(
                mesh, V, F, start_p=p, start_face=f, target_vid=src_idx, X=X, Y=Y
            )
            if len(finish_pts) > 1:
                pts.extend(finish_pts[1:]); total_len += finish_len
                hit_source = True
            break

        pts.append(q.copy())
        total_len += seg_len

        # cross consistently to neighbor using barycentric transfer
        lam_old = barycentric_coords(q, a, b, c)
        nb_f, nb_e = neighbors[f, l_edge]
        if nb_f < 0:
            # boundary – finish straight
            finish_pts, finish_len = finish_segment_to_vertex(
                mesh, V, F, start_p=q, start_face=f, target_vid=src_idx, X=X, Y=Y
            )
            if len(finish_pts) > 1:
                pts.extend(finish_pts[1:]); total_len += finish_len
                hit_source = True
            break

        vids_nb = F[nb_f]
        # map the two shared vertices' weights to neighbor local ordering
        try:
            a_nb = int(np.where(vids_nb == gA)[0][0])
            b_nb = int(np.where(vids_nb == gB)[0][0])
        except IndexError:
            a_nb = int(np.where(vids_nb == gB)[0][0])
            b_nb = int(np.where(vids_nb == gA)[0][0])
            sA_old, sB_old = sB_old, sA_old

        lam_new = np.zeros(3, float)
        lam_new[a_nb] = lam_old[sA_old]
        lam_new[b_nb] = lam_old[sB_old]
        c_nb = 3 - a_nb - b_nb
        lam_new[c_nb] = eps_bary
        lam_new /= lam_new.sum()

        a2, b2, c2 = V[vids_nb[0]], V[vids_nb[1]], V[vids_nb[2]]
        p = lam_new[0]*a2 + lam_new[1]*b2 + lam_new[2]*c2
        f = int(nb_f)

    # ---- If the face tracer failed to hit the source, fall back to metric Dijkstra
    if not hit_source:
        if verbose: print("[trace] falling back to metric Dijkstra...")
        # reconstruct G2 from Ginv2
        G2 = np.linalg.inv(Ginv2)
        v_start = int(nearest_vertex(V, p))
        seed_pts, seed_len = dijkstra_metric_path(V, F, G2, X, Y, src=v_start, tgt=int(src_idx))
        return seed_pts, float(seed_len), False

    return np.vstack(pts), float(total_len), True





# --------------------------
# Main
# --------------------------
def main():
    obj_path = Path("C:/Users/jdkg7/Documents/Python/Metric Modification/torus.obj")

    # Load & primitives
    mesh, V, F = load_mesh(obj_path)
    A, grads, X, Y, h = build_primitives(mesh, V, F)

    # ---- User metric (per-face 2x2 SPD in frame (X,Y))
    # Example 1: None -> identity metric (Euclidean case)
    G2 = None

    # Example 2: mild anisotropy aligned with X (uncomment to try)
    # m = len(F)
    # G2 = np.repeat(np.eye(2)[None,:,:], m, axis=0)
    # G2[:,0,0] = 2.0   # stretch in X (lengths larger)
    # G2[:,1,1] = 1.0

    # Assemble operators in the chosen metric
    K, M, Ginv2, det_sqrt = assemble_operators(V, F, A, grads, X, Y, G2)

    # Pick rough source & target, project to surface, then choose nearest vertices
    src_rough = np.array([0.8, -0.2, -0.])
    tgt_rough = np.array([-1.1, -0.5, -0.3])
    obs_rough = np.array([0.1, -0.9, -0.1])
    prox = trimesh.proximity.ProximityQuery(mesh)
    src_proj = (prox.on_surface([src_rough])[0])[0]
    tgt_proj = (prox.on_surface([tgt_rough])[0])[0]
    obs_proj = (prox.on_surface([obs_rough])[0])[0]
    src = nearest_vertex(V, src_proj)
    tgt = nearest_vertex(V, tgt_proj)
    obs = nearest_vertex(V, obs_proj)

    # Heat distance (auto-tuned t)
    dist, t_best, _ = heat_distance_auto_metric(V, F, A, grads, X, Y, K, M, Ginv2, det_sqrt, src, h)
    d_tgt, _ = distance_at_point(mesh, V, F, dist, tgt_rough, prox=prox)

    print(f"Loaded mesh: {len(V)} vertices, {len(F)} faces; watertight={mesh.is_watertight}")
    print(f"Sum(M) ≈ {M.diagonal().sum():.6f}")
    print(f"Source vertex: {src}, Target vertex: {tgt}")
    print(f"\n[Chosen] t = {t_best:.6e}")
    print("\n---- Results ----")
    print(f"Distance(target) ≈ {d_tgt:.6f}")
    print(f"min(dist)={float(dist.min()):.6f}, max(dist)={float(dist.max()):.6f}")

    # Trace metric geodesic (steepest descent of φ in metric)
    print("\nTracing metric geodesic path...")
    path_pts, path_len, hit_src = trace_geodesic_polyline_metric(
        mesh, V, F, dist, grads, X, Y, Ginv2, start_point=tgt_proj, src_idx=src
    )
    print(f"Polyline length ≈ {path_len:.6f}, hit_source={hit_src}")

    # Visualization
    dist_norm = 1.0 - (dist - dist.min()) / (dist.max() - dist.min() + 1e-16)
    cmap = plt.get_cmap('coolwarm')
    face_colors = cmap(dist_norm[F].mean(axis=1))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1]); ax.axis('off')
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=face_colors, linewidths=0.05, edgecolors='none'))
    ax.scatter(*src_proj, color='lime', s=40, label='Source')
    ax.scatter(*tgt_proj, color='cyan', s=40, label='Target')
    ax.plot(path_pts[:,0], path_pts[:,1], path_pts[:,2], color='k', linewidth=2.0, alpha=0.95, label='Geodesic')
    ax.legend(loc='upper right'); ax.view_init(elev=30, azim=30)
    # plt.show()

    # Interactive color preview
    colors = (plt.get_cmap('coolwarm')(1.0 - (dist - dist.min()) / (dist.max() - dist.min() + 1e-16))[:, :3] * 255).astype(np.uint8)
    mesh.visual.vertex_colors = colors

    # source/target markers
    src_ball = small_sphere(src_proj, radius=0.02, rgba=(255, 0, 0, 255))   # red
    tgt_ball = small_sphere(tgt_proj, radius=0.02, rgba=(0, 0, 255, 255))   # blue
    obs_ball = small_sphere(obs_proj, radius=0.02, rgba=(0, 255, 0, 255))   # green

    # polyline entity
    path_obj = path3d_from_polyline(path_pts, color=(0, 0, 0, 255))         # black

    # assemble and show scene
    scene = trimesh.Scene([mesh, path_obj, src_ball, tgt_ball, obs_ball])
    scene.show()


if __name__ == "__main__":
    main()