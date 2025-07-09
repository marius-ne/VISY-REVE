import numpy as np
import plotly.offline as offline
import plotly.graph_objects as go
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import json
import trimesh
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import mpl_toolkits


def plot_camera_frustums_matplotlib(
    poses: List[np.ndarray],
    pose_names: Optional[List[str]] = None,
    fov_y: float = 60.0,
    aspect: float = 4/3,
    z_near: float = 0.1,
    z_far: float = 1.0,
    scale: Union[float,List[float]] = 1.0,
    colors: Optional[List[str]] = None,
    line_width: float = 2.0,
    show_labels: bool = True,
    mesh: Optional[trimesh.Trimesh] = None,
    mesh_color: str = "lightgrey",
    mesh_colormap: str = "viridis",
    mesh_opacity: float = 0.5,
    mesh_scale: float = 1.0,
    elev: float = 30.0,
    azim: float = -60.0,
    figsize: tuple = (10, 8),
    return_fig: bool = False,
    ax: plt.Axes = None,
    camera_center_size: float = 30
) -> plt.Figure:
    """
    Static Matplotlib 3D plot of camera poses + frustums and optional mesh.

    All parameters mirror the Plotly version, plus:
      elev, azim : floats
          Elevation and azimuth (in degrees) for the 3D view.
      figsize : tuple
          Figure size.
      return_fig: bool
            If True (False by default) the figure will not be shown but returned as 
            an object.
      ax: plt.Axes
            (optional) Axes on which to draw the graph. If not provided, new figure
            will be created.
      camera_center_size: float
            Size of the scatter points of the camera centers for each frustum. 
            Default is 30.

    Returns
    -------
    fig : plt.Figure
        The matplotlib Figure.
    """
    # Validate
    N = len(poses)
    if pose_names is not None and len(pose_names) != N:
        raise ValueError(f"pose_names length ({len(pose_names)}) must match number of poses ({N})")
    # Convert passive (world2cam) to active (cam2world)
    poses = [np.linalg.inv(p) for p in poses]

    # Compute half‐width/height at unit distance
    fov_y_rad = np.deg2rad(fov_y)
    half_h = np.tan(fov_y_rad / 2.0)
    half_w = half_h * aspect

    # Prep colors: if none provided, use a single default color and disable labels
    default_color = 'blue'
    if colors is None:
        colors = [default_color]
    # mesh vertices/faces if given
    if mesh is not None:
        verts = np.asarray(mesh.vertices) * mesh_scale
        faces = np.asarray(mesh.faces)
    
    # Never show camera labels  
    show_labels = False

    # Create or use provided axis
    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={'projection': '3d'}
        )
    else:
        fig = ax.figure
        # If they passed a regular 2D Axes, replace it with a 3D Axes in the same location:
        if not isinstance(ax, mpl_toolkits.mplot3d.Axes3D):
            spec = ax.get_subplotspec()
            fig.delaxes(ax)
            ax = fig.add_subplot(spec, projection='3d')

    # Now it's safe to call view_init
    ax.view_init(elev=elev, azim=azim)

    # Plot mesh first (semi‐transparent), colored by vertex Z-height
    if mesh is not None:
        verts = np.asarray(mesh.vertices) * mesh_scale
        faces = np.asarray(mesh.faces)
        # per-vertex heights
        zs = verts[:, 2]
        # normalize to [0,1]
        norm = plt.Normalize(vmin=zs.min(), vmax=zs.max())
        cmap = plt.get_cmap(mesh_colormap)
        # compute a color per face: mean of its vertices' normalized Z
        face_z = zs[faces].mean(axis=1)
        face_colors = cmap(norm(face_z))
        # build the Poly3DCollection
        mesh_verts = [verts[f] for f in faces]
        poly = Poly3DCollection(mesh_verts, facecolors=face_colors,
                                edgecolor='none', alpha=mesh_opacity)
        ax.add_collection3d(poly)       

    # Plot origin
    ax.scatter([0], [0], [0], color='red', s=40, label='origin')

    # --- New: allow `scale` to be a per‐pose array or a single scalar ---
    if np.isscalar(scale):
        scales = [scale] * N
    else:
        scales = list(scale)
        if len(scales) != N:
            raise ValueError(f"scale array length ({len(scales)}) must match number of poses ({N})")
        
    for i, pose in enumerate(poses):
        if pose.shape != (4,4):
            raise ValueError(f"Pose #{i} has shape {pose.shape}; expected (4,4).")

        # camera center in world coords
        center = (pose @ np.array([0,0,0,1]))[:3]

        # plane half‐sizes
        scale = scales[i]
        hw_near, hh_near = half_w * z_near * scale, half_h * z_near * scale
        hw_far,  hh_far  = half_w * z_far  * scale, half_h * z_far  * scale

        # corners in camera‐space
        nc = np.array([
            [-hw_near, -hh_near, z_near*scale],
            [ hw_near, -hh_near, z_near*scale],
            [ hw_near,  hh_near, z_near*scale],
            [-hw_near,  hh_near, z_near*scale],
        ])
        fc = np.array([
            [-hw_far, -hh_far, z_far*scale],
            [ hw_far, -hh_far, z_far*scale],
            [ hw_far,  hh_far, z_far*scale],
            [-hw_far,  hh_far, z_far*scale],
        ])

        # transform to world
        pts_cam = np.vstack([nc, fc])
        homog = np.hstack([pts_cam, np.ones((8,1))])
        pts_world = (pose @ homog.T).T[:, :3]

        # choose color
        col = colors[i % len(colors)]

        # plot camera center
        ax.scatter(*center, color=col, s=camera_center_size)
        if show_labels:
            name = pose_names[i] if pose_names else f"cam_{i}"
            ax.text(*center, name, color=col)

        # build edge list
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]
        segs = [(pts_world[u], pts_world[v]) for u,v in edges]
        lc = Line3DCollection(segs, colors=col, linewidths=line_width)
        ax.add_collection3d(lc)

    # Set labels & equal aspect
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Equalize axes
    all_pts = np.vstack([ [0,0,0] ] + [ (pose @ np.array([0,0,0,1]))[:3] for pose in poses ])
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2.0
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

    #ax.legend()
    fig.tight_layout()
    if return_fig:
        return fig
    else:   
        fig.show()


def plot_camera_frustums(
    poses: List[np.ndarray],
    pose_names: Optional[List[str]] = None,
    fov_y: float = 60.0,
    aspect: float = 4/3,
    z_near: float = 0.1,
    z_far: float = 1,
    scale: float = 1.0,
    colors: Optional[List[str]] = None,
    line_width: float = 2.0,
    show_labels: bool = True,
    mesh: Optional[trimesh.Trimesh] = None,
    mesh_color: str = "lightgrey",
    mesh_colormap: str = "Viridis",
    mesh_opacity: float = 0.5,
    mesh_scale: float = 1,
    inline: bool = True
) -> go.Figure:
    """
    Plot multiple camera poses in 3D, each with a wireframe frustum, and optionally label each pose.
    Also adds a red point at the origin.

    Parameters
    ----------
    poses : List[np.ndarray]
        A list of N camera poses. Each pose must is passive world2cam (translation vector
        in camera coordinates).

    pose_names : Optional[List[str]], default=None
        A list of N strings, naming each pose. If provided, its length must match len(poses).
        These names will appear in the legend next to each camera's colored marker.

    fov_y : float, default=60.0
        Vertical field of view in degrees.

    aspect : float, default=4/3
        Width / height ratio of the image plane.

    z_near : float, default=0.2
        Distance to the near plane in camera‐space.

    z_far : float, default=1.0
        Distance to the far plane in camera‐space.

    scale : float, default=1.0
        Global scale factor for the frusta.

    colors : Optional[List[str]], default=None
        List of color strings (e.g. ["red", "green", "blue", ...]). If None, Plotly’s default palette is used.
        If provided, its length should be at least len(poses), or colors will cycle.

    line_width : float, default=2.0
        Thickness of each frustum edge.

    Returns
    -------
    fig : go.Figure
        A Plotly Figure containing all camera centers (with labels) and their frustums,
        plus a red point at the origin.
    """
    N = len(poses)
    if pose_names is not None and len(pose_names) != N:
        raise ValueError(f"pose_names length ({len(pose_names)}) must match number of poses ({N})")
    
    # This function was written for active poses but we expect
    # passive world2cam. Therefore we invert
    poses = [np.linalg.inv(pose) for pose in poses]

    # Precompute image‐plane half‐width and half‐height at z=1
    fov_y_rad = np.deg2rad(fov_y)
    half_h = np.tan(fov_y_rad / 2.0)
    half_w = half_h * aspect

    # Prepare the Figure
    fig = go.Figure()

    # 0) If mesh provided, draw it first
    if mesh is not None:
        verts = np.asarray(mesh.vertices) * mesh_scale
        faces = np.asarray(mesh.faces)

        # If the user didn’t pass a scalar field, default to z‐height:
        mesh_scalar = verts[:, 2].copy()

        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                intensity=mesh_scalar,
                colorscale=mesh_colormap,
                showscale=True,
                cmin=mesh_scalar.min(),
                cmax=mesh_scalar.max(),
                opacity=mesh_opacity,
                name="mesh",
                showlegend=False
            )
        )

    # 1) Add a red point at the origin (0,0,0)
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode='markers',
            marker=dict(size=6, color='red'),
            name='origin'
        )
    )

    # If no colors provided, let Plotly assign default colors
    if colors is None:
        colors = []

    for i, pose in enumerate(poses):
        if pose.shape != (4, 4):
            raise ValueError(f"Pose #{i} has shape {pose.shape}; expected (4,4).")

        # Camera center in world coordinates
        cam_center_world = (pose @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]

        # Compute near/far plane half‐sizes
        hw_near = half_w * z_near
        hh_near = half_h * z_near
        hw_far  = half_w * z_far
        hh_far  = half_h * z_far

        # Frustum corners in camera-space
        ncorners = np.array([
            [-hw_near, -hh_near, z_near],  # near: bottom-left
            [ hw_near, -hh_near, z_near],  # near: bottom-right
            [ hw_near,  hh_near, z_near],  # near: top-right
            [-hw_near,  hh_near, z_near],  # near: top-left
        ])
        fcorners = np.array([
            [-hw_far, -hh_far, z_far],     # far: bottom-left
            [ hw_far, -hh_far, z_far],     # far: bottom-right
            [ hw_far,  hh_far, z_far],     # far: top-right
            [-hw_far,  hh_far, z_far],     # far: top-left
        ])

        # Apply scale
        ncorners *= scale
        fcorners *= scale

        # Transform all 8 corners into world space
        world_pts = []
        for corner in np.vstack([ncorners, fcorners]):
            cam_pt_homog = np.hstack([corner, 1.0])
            world_pt = (pose @ cam_pt_homog)[:3]
            world_pts.append(world_pt)
        world_pts = np.vstack(world_pts)  # shape (8, 3)

        # Determine color for this camera
        color = colors[i % len(colors)] if colors else None

        # 2) Draw camera center with label (pose_names[i] or default "cam_i")
        label = pose_names[i] if pose_names is not None else f"cam_{i}"
        if show_labels:
            mode = 'markers+text'
            text = [label]
            textpos = 'top center'
            textfont = dict(color=color or 'black')
        else:
            mode = 'markers'
            text = None
            textpos = None
            textfont = None
        fig.add_trace(
            go.Scatter3d(
                x=[cam_center_world[0]],
                y=[cam_center_world[1]],
                z=[cam_center_world[2]],
                mode=mode,
                marker=dict(size=(5*scale)/0.5, color=color or 'black'),
                text=text,
                textposition=textpos,
                textfont=textfont,
                name=label,
                showlegend=True
            )
        )

        # 3) Draw frustum edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # near‐plane rectangle
            (4, 5), (5, 6), (6, 7), (7, 4),  # far‐plane rectangle
            (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
        ]
        for (u, v) in edges:
            x_coords = [world_pts[u, 0], world_pts[v, 0]]
            y_coords = [world_pts[u, 1], world_pts[v, 1]]
            z_coords = [world_pts[u, 2], world_pts[v, 2]]

            fig.add_trace(
                go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='lines',
                    line=dict(color=color or 'blue', width=line_width),
                    name=f"{label}_edge_{u}_{v}",
                    showlegend=False
                )
            )

    # Final layout tweaks
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', backgroundcolor="rgb(230, 230, 230)"),
            yaxis=dict(title='Y', backgroundcolor="rgb(230, 230, 230)"),
            zaxis=dict(title='Z', backgroundcolor="rgb(230, 230, 230)"),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=True,
        title="Camera Poses with Frustums (Origin in Red)"
    )
    if not inline:
        html_path = offline.plot(fig, auto_open=True, filename="media/plotly/frustums.html")
    else:
        fig.show()

def plot_poses(poses, 
               on_unit_sphere=False,
               sphere_resolution=50, 
               colors=None,
               sphere_opacity=0.5,
               inline = True):
    """
    Visualize a list of poses as points, optionally on the unit sphere,
    by plotting each poses camera center.

    Args:
        poses (np.ndarray): 4x4 passive w2c poses.
        sphere_resolution (int): Number of subdivisions for the spherical mesh
            (higher → smoother sphere).
        colors (Optional[List[str]]): List of color strings for each pose. If None, all points are red.
        inline: If False (True by default) then saved as html under "media/plotly/poses.html"

    Returns:
        fig (plotly.graph_objects.Figure): A Plotly 3D figure showing the unit sphere
            and the rotation‐axis points.
    """
    # 1. For each rotation matrix, extract the axis vector (unit length).
    axes = []
    for pose in poses:
        rot = Rotation.from_matrix(pose[:3,:3])
        tvec = pose[:3,3]
        axis = -rot.inv().apply(tvec)
        if on_unit_sphere:
            axis /= np.linalg.norm(axis)
            axis *= 1.05 # extend slightly, otherwise there is aliasing
        axes.append(axis)

    if len(axes) == 0:
        raise ValueError("No non‐zero rotations found to plot.")

    axes = np.vstack(axes)  # shape (M, 3), M ≤ N

    # 2. Build a unit‐sphere mesh (for background)
    u = np.linspace(0, np.pi, sphere_resolution)
    v = np.linspace(0, 2 * np.pi, sphere_resolution)
    u, v = np.meshgrid(u, v)
    x_sphere = np.sin(u) * np.cos(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(u)

    # 3. Create Plotly figure with:
    #    - a semi‐transparent unit sphere surface
    #    - a scatter3d of all axes points

    sphere_surface = go.Surface(
        x = x_sphere,
        y = y_sphere,
        z = z_sphere,
        colorscale="Blues",
        opacity=sphere_opacity,
        showscale=False,
        hoverinfo="skip"
    )

    # Handle colors for each pose
    if colors is not None:
        if len(colors) != len(axes):
            raise ValueError(f"Length of colors ({len(colors)}) must match number of poses ({len(axes)})")
        # Plot each axis as a separate point with its color
        scatter_axes = go.Scatter3d(
            x = axes[:, 0],
            y = axes[:, 1],
            z = axes[:, 2],
            mode = "markers",
            marker = dict(
                size=4,
                color=colors,
                opacity=0.8
            ),
            name = "Rotation axes"
        )
    else:
        scatter_axes = go.Scatter3d(
            x = axes[:, 0],
            y = axes[:, 1],
            z = axes[:, 2],
            mode = "markers",
            marker = dict(
                size=4,
                color="red",
                opacity=0.8
            ),
            name = "Rotation axes"
        )

    if on_unit_sphere:
        layout = go.Layout(
            scene = dict(
                xaxis = dict(
                    title="X",
                    range=[-1.1, 1.1],
                    showbackground=True,
                    showgrid=True  # Enable grid lines
                ),
                yaxis = dict(
                    title="Y",
                    range=[-1.1, 1.1],
                    showbackground=True,
                    showgrid=True  # Enable grid lines
                ),
                zaxis = dict(
                    title="Z",
                    range=[-1.1, 1.1],
                    showbackground=True,
                    showgrid=True  # Enable grid lines
                ),
                aspectmode = "manual",
                aspectratio = dict(x=1, y=1, z=1)
            ),
            margin = dict(l=0, r=0, b=0, t=0),
            title = "Rotation Axes on Unit Sphere"
        )
    else:
        # Get maximum extend of the poses on each axes and choose maximum
        max_range = max([max(axes[:,axis]) for axis in range(3)])
        max_range *= 1.1 # extend slightly
        axes_range = [-max_range,max_range]
        layout = go.Layout(
        scene = dict(
            xaxis = dict(
                title="X",
                range=axes_range,
                showbackground=True,
                showgrid=True  # Enable grid lines
            ),
            yaxis = dict(
                title="Y",
                range=axes_range,
                showbackground=True,
                showgrid=True  # Enable grid lines
            ),
            zaxis = dict(
                title="Z",
                range=axes_range,
                showbackground=True,
                showgrid=True  # Enable grid lines
            ),
            aspectmode = "manual",
            aspectratio = dict(x=1, y=1, z=1)
        ),
        margin = dict(l=0, r=0, b=0, t=0),
        title = "Rotation Axes on Unit Sphere"
    )

    fig = go.Figure(data=[sphere_surface, scatter_axes], layout=layout)
    if not inline:
        html_path = offline.plot(fig, auto_open=True, filename="media/plotly/poses.html")
    else:
        fig.show()


def plot_poses_with_cone(
    poses,
    on_unit_sphere=False,
    sphere_resolution=50,
    colors=None,
    sphere_opacity=0.5,
    # new cone parameters:
    cone_endpoint=None,        # array-like length-3, e.g. [x,y,z]
    cone_radius=None,          # float
    cone_color="blue",         # any Plotly color
    cone_resolution=20         # number of segments around the base
):
    """
    Visualize poses as points (and optionally a unit sphere background),
    and—if requested—draw an opaque cone from the origin out to `cone_endpoint`
    with radius `cone_radius` at its base, plus mark the endpoint as a lime dot.

    Args:
        poses (np.ndarray): list of 4×4 passive w2c poses.
        on_unit_sphere (bool): normalize axes to lie on (just outside) unit sphere.
        sphere_resolution (int): subdivisions for the unit-sphere mesh.
        colors (List[str] or None): per-pose colors; defaults to red.
        sphere_opacity (float): opacity of the sphere background.
        cone_endpoint (array-like or None): [x, y, z] tip of the cone.
        cone_radius (float or None): radius of the cone at `cone_endpoint`.
        cone_color (str): color of the cone.
        cone_resolution (int): number of sides for the cone’s circular base.

    Returns:
        fig (plotly.graph_objects.Figure)
    """
    # --- compute axis points from poses ---
    axes = []
    for pose in poses:
        rot = Rotation.from_matrix(pose[:3, :3])
        tvec = pose[:3, 3]
        axis = -rot.inv().apply(tvec)
        if on_unit_sphere:
            axis /= np.linalg.norm(axis)
            axis *= 1.05
        axes.append(axis)
    if not axes:
        raise ValueError("No poses to plot.")
    axes = np.vstack(axes)

    # --- unit-sphere mesh ---
    u = np.linspace(0, np.pi, sphere_resolution)
    v = np.linspace(0, 2 * np.pi, sphere_resolution)
    u, v = np.meshgrid(u, v)
    x_s = np.sin(u) * np.cos(v)
    y_s = np.sin(u) * np.sin(v)
    z_s = np.cos(u)
    sphere = go.Surface(
        x=x_s, y=y_s, z=z_s,
        colorscale="Blues",
        opacity=sphere_opacity,
        showscale=False,
        hoverinfo="skip",
    )

    # --- scatter of axes ---
    marker_kwargs = dict(size=4, opacity=1)
    if colors is not None:
        if len(colors) != len(axes):
            raise ValueError(f"colors length {len(colors)} ≠ poses {len(axes)}")
        marker_kwargs["color"] = colors
    else:
        marker_kwargs["color"] = "red"

    scatter = go.Scatter3d(
        x=axes[:, 0], y=axes[:, 1], z=axes[:, 2],
        mode="markers",
        marker=marker_kwargs,
        name="Rotation axes"
    )

    # --- optional cone mesh + endpoint marker ---
    data = [sphere, scatter]

    if cone_endpoint is not None and cone_radius is not None:
        ep = np.asarray(cone_endpoint, dtype=float)
        r = float(cone_radius)

        # -- build cone geometry --
        L = np.linalg.norm(ep)
        if L == 0:
            raise ValueError("cone_endpoint must not be the origin")
        direction = ep / L
        arb = np.array([0, 0, 1.0]) if abs(direction[2]) < 0.99 else np.array([0, 1.0, 0])
        u_vec = np.cross(arb, direction)
        u_vec /= np.linalg.norm(u_vec)
        v_vec = np.cross(direction, u_vec)

        thetas = np.linspace(0, 2 * np.pi, cone_resolution, endpoint=False)
        circle_pts = np.stack([
            ep + r * (np.cos(t) * u_vec + np.sin(t) * v_vec)
            for t in thetas
        ], axis=0)

        verts = np.vstack((np.zeros((1, 3)), circle_pts))
        idxs = []
        for i in range(1, cone_resolution + 1):
            j = i + 1 if i < cone_resolution else 1
            idxs.append((0, i, j))
        i_idx, j_idx, k_idx = zip(*idxs)

        cone_mesh = go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=i_idx, j=j_idx, k=k_idx,
            color=cone_color,
            opacity=1,
            name="Cone"
        )
        data.append(cone_mesh)

        # -- highlight endpoint --
        endpoint_scatter = go.Scatter3d(
            x=[ep[0]], y=[ep[1]], z=[ep[2]],
            mode="markers",
            marker=dict(size=5, color="lime", opacity=1.0),
            name="Cone endpoint"
        )
        data.append(endpoint_scatter)

    # --- layout ---
    if on_unit_sphere:
        axis_range = [-1.1, 1.1]
    else:
        mr = axes.max() * 1.1
        axis_range = [-mr, mr]
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title="X", range=axis_range, showbackground=True, showgrid=True),
            yaxis=dict(title="Y", range=axis_range, showbackground=True, showgrid=True),
            zaxis=dict(title="Z", range=axis_range, showbackground=True, showgrid=True),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Rotation Axes" + (" on Unit Sphere" if on_unit_sphere else "")
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()
    # return fig


def plot_rotations_on_sphere(rotation_matrices, sphere_resolution=50):
    """
    Visualize a list of 3×3 rotation matrices as points on the unit sphere
    by plotting each rotation's true rotation‐axis (unit vector) on S^2.

    For each R ∈ SO(3), this function finds the “axis” (i.e., eigenvector with eigenvalue 1),
    normalizes it to unit length, and places that point on the sphere.  The sphere itself is plotted
    semi‐transparently for reference.

    Args:
        rotation_matrices (Iterable[np.ndarray]): 
            An iterable of shape-(3,3) numpy arrays, each a valid rotation matrix in SO(3).

        sphere_resolution (int):
            Number of subdivisions for the spherical mesh.  Larger → smoother sphere.

    Returns:
        fig (plotly.graph_objects.Figure):
            A Plotly 3D figure containing:
              - A semi‐transparent unit‐sphere surface
              - Red markers at each rotation’s axis‐direction on S^2

    Raises:
        ValueError: 
            - If `rotation_matrices` is empty.
            - If any matrix is not 3×3 or not close to a valid rotation matrix.
    """
    if len(rotation_matrices) == 0:
        raise ValueError("`rotation_matrices` is empty.")

    # 1. Extract and normalize the axis‐of‐rotation for each 3×3 matrix
    axes = []
    for idx, mat in enumerate(rotation_matrices):
        mat = np.asarray(mat)
        if mat.shape != (3, 3):
            raise ValueError(f"Rotation at index {idx} is not 3×3, got shape {mat.shape}.")

        # Convert to axis–angle (rotvec); as_rotvec() returns (axis * angle)
        rot = Rotation.from_matrix(mat)
        rotvec = rot.as_rotvec()  # shape = (3,)
        angle = np.linalg.norm(rotvec)

        # If angle ≈ 0, matrix is identity → axis is arbitrary (pick x‐axis)
        if angle < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = rotvec / angle  # unit‐length axis

        axes.append(axis)

    axes = np.vstack(axes)  # shape (N, 3)

    # 2. Build a unit‐sphere mesh for the background
    theta = np.linspace(0, np.pi, sphere_resolution)
    phi   = np.linspace(0, 2 * np.pi, sphere_resolution)
    θ, φ  = np.meshgrid(theta, phi)
    x_sphere = np.sin(θ) * np.cos(φ)
    y_sphere = np.sin(θ) * np.sin(φ)
    z_sphere = np.cos(θ)

    sphere_surface = go.Surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        colorscale="Blues",
        opacity=0.4,
        showscale=False,
        hoverinfo="skip",
    )

    # 3. Plot the axes as red markers on the sphere
    scatter_axes = go.Scatter3d(
        x=axes[:, 0],
        y=axes[:, 1],
        z=axes[:, 2],
        mode="markers",
        marker=dict(size=4, color="red", opacity=0.8),
        name="Rotation axes"
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title="X", range=[-1.1, 1.1], showbackground=False),
            yaxis=dict(title="Y", range=[-1.1, 1.1], showbackground=False),
            zaxis=dict(title="Z", range=[-1.1, 1.1], showbackground=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        title="Rotation Axes on the Unit Sphere",
    )

    fig = go.Figure(data=[sphere_surface, scatter_axes], layout=layout)
    fig.show()
    #return fig


def plot_points_on_sphere(vectors, sphere_resolution=50):
    """
    Plot a collection of 3D vectors as points on the unit sphere by normalizing each vector.

    Args:
        vectors (Iterable[np.ndarray] or np.ndarray):
            An iterable or array of shape (N, 3), each row a 3D vector.  Must have at least one vector.
        sphere_resolution (int):
            Number of subdivisions for the spherical mesh (larger → smoother sphere).

    Returns:
        fig (plotly.graph_objects.Figure):
            A Plotly 3D figure showing:
              - A semi‐transparent unit‐sphere surface
              - Blue markers at each normalized vector direction on S²

    Raises:
        ValueError:
            - If `vectors` is empty.
            - If any vector does not have shape (3,).
            - If any vector has (near) zero length.
    """
    # Convert to numpy array and validate
    arr = np.asarray(vectors)
    if arr.ndim == 1:
        # Single vector case: reshape to (1, 3)
        if arr.size != 3:
            raise ValueError(f"Expected vector of length 3, got shape {arr.shape}.")
        arr = arr.reshape(1, 3)
    elif arr.ndim == 2:
        if arr.shape[1] != 3:
            raise ValueError(f"Each vector must have 3 components, got shape {arr.shape}.")
    else:
        raise ValueError(f"Expected input of shape (N, 3) or (3,), got {arr.shape}.")

    N = arr.shape[0]
    if N == 0:
        raise ValueError("`vectors` is empty.")

    # Normalize each vector to unit length
    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("One or more vectors have near-zero length and cannot be normalized.")
    unit_dirs = (arr.T / norms).T  # shape (N, 3)

    # Build unit‐sphere mesh for the background
    theta = np.linspace(0, np.pi, sphere_resolution)
    phi   = np.linspace(0, 2 * np.pi, sphere_resolution)
    θ, φ  = np.meshgrid(theta, phi)
    x_sphere = np.sin(θ) * np.cos(φ)
    y_sphere = np.sin(θ) * np.sin(φ)
    z_sphere = np.cos(θ)

    sphere_surface = go.Surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        colorscale="Greys",
        opacity=0.3,
        showscale=False,
        hoverinfo="skip"
    )

    # Plot the normalized vectors as blue markers
    scatter_points = go.Scatter3d(
        x=unit_dirs[:, 0],
        y=unit_dirs[:, 1],
        z=unit_dirs[:, 2],
        mode="markers",
        marker=dict(size=4, color="blue", opacity=0.8),
        name="Normalized points"
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title="X", range=[-1.1, 1.1], showbackground=False),
            yaxis=dict(title="Y", range=[-1.1, 1.1], showbackground=False),
            zaxis=dict(title="Z", range=[-1.1, 1.1], showbackground=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        title="Points on Unit Sphere"
    )

    fig = go.Figure(data=[sphere_surface, scatter_points], layout=layout)
    fig.show()
    #return fig

def plot_mesh_with_diameter_and_sphere(mesh, 
                                       endpoints, 
                                       max_dist, 
                                       mesh_opacity=1.0, 
                                       sphere_resolution=30,
                                       radius_margin=0.1):
    """
    Plot a trimesh Trimesh object together with its diameter endpoints, connecting line,
    and the minimal enclosing sphere whose center is the midpoint of the diameter.

    Args:
        mesh (trimesh.Trimesh):
            The original mesh to plot.
        endpoints (np.ndarray):
            A (2, 3) array of the two farthest-apart points (from compute_mesh_diameter).
        max_dist (float):
            The diameter length (distance between the two endpoints).
        mesh_opacity (float, optional):
            Opacity for the mesh itself. Default = 1.0.
        sphere_resolution (int, optional):
            Number of subdivisions for creating the sphere surface. Default = 30.

    Returns:
        fig (plotly.graph_objects.Figure):
            A Plotly 3D figure showing:
              - The original mesh as a Mesh3d trace.
              - Two red markers at the farthest‐apart endpoints.
              - A black line segment connecting those two points.
              - A semi‐transparent sphere (centered at the midpoint) with radius = max_dist/2.
    """
    # 1) Original mesh trace
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    mesh_trace = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name="Mesh",
        opacity=mesh_opacity,
        color="lightblue",
        flatshading=True
    )

    # 2) Endpoints as red markers
    scatter_endpoints = go.Scatter3d(
        x=endpoints[:, 0],
        y=endpoints[:, 1],
        z=endpoints[:, 2],
        mode="markers",
        marker=dict(size=5, color="red"),
        name="Farthest points"
    )

    # 3) Line segment connecting the two endpoints
    p_i, p_j = endpoints
    line_segment = go.Scatter3d(
        x=[p_i[0], p_j[0]],
        y=[p_i[1], p_j[1]],
        z=[p_i[2], p_j[2]],
        mode="lines",
        line=dict(color="black", width=3),
        name=f"Diameter = {max_dist:.3f}"
    )

    # 4) Compute sphere center and radius
    center = (p_i + p_j) / 2.0
    radius = (max_dist / 2.0)*radius_margin

    # 5) Build sphere surface mesh around (center, radius)
    theta = np.linspace(0, np.pi, sphere_resolution)
    phi   = np.linspace(0, 2 * np.pi, sphere_resolution)
    θ, φ  = np.meshgrid(theta, phi)
    x_sphere = center[0] + radius * np.sin(θ) * np.cos(φ)
    y_sphere = center[1] + radius * np.sin(θ) * np.sin(φ)
    z_sphere = center[2] + radius * np.cos(θ)

    sphere_surface = go.Surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        colorscale="Greys",
        opacity=0.3,
        showscale=False,
        hoverinfo="skip",
        name="Enclosing sphere"
    )

    # 6) Determine axis ranges with padding
    all_pts = np.concatenate((verts, endpoints, np.vstack((x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten())).T), axis=0)
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()
    pad = 0.05 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_range = [x_min - pad, x_max + pad]
    y_range = [y_min - pad, y_max + pad]
    z_range = [z_min - pad, z_max + pad]

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title="X", range=x_range, showbackground=False),
            yaxis=dict(title="Y", range=y_range, showbackground=False),
            zaxis=dict(title="Z", range=z_range, showbackground=False),
            aspectmode="manual",
            aspectratio=dict(
                x=(x_range[1] - x_range[0]),
                y=(y_range[1] - y_range[0]),
                z=(z_range[1] - z_range[0])
            )
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"Mesh Diameter: {max_dist:.3f} and Enclosing Sphere"
    )

    fig = go.Figure(data=[mesh_trace, scatter_endpoints, line_segment, sphere_surface], layout=layout)
    fig.show()
    #return fig

def plot_mc_results(json_file: str,
                    fx: float,
                    x_vars: List[str],
                    metrics_list: List[str],
                    dims: List[str] = ['2D','3D'],
                    log_x: bool = False,
                    log_y: bool = False,
                    show_corr: bool = False):
    """
    Reads a JSON file and plots each requested metric and dimension against each x-variable.

    Args:
        json_file (str): Path to the JSON file.
        fx (float): Camera focal length (in pixels), needed for VBN if requested.
        x_vars (List[str]): keys to use as x-axes (e.g. ['cdist','bdd']).
        metrics_list (List[str]):
            Metric prefixes to plot.  
            For each `m`, we look for entry[f"{m}2D"] and entry[f"{m}3D"],
            unless 'vbn' in which case we compute vbnHOM/vbn3DT on the fly.
        dims (List[str]): which dimensions to include; subset of ['2D','3D'].
        log_x (bool): If True, log scale on x-axes.
        log_y (bool): If True, log scale on y-axes.
        show_corr (bool): If True, display the Pearson r in the title.
    """
    # 1) Load
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 2) Extract x_vars
    extracted = {x: np.array([entry[x] for entry in data]) for x in x_vars}
    if 'cdist' and 'adist' in x_vars:
        x_vars.append('comb')
        extracted['comb'] = extracted['cdist'] + extracted['adist']

    # 3) Extract standard metrics
    for m in metrics_list:
        if m != 'vbn':
            for d in ['2D','3D']:
                extracted[f'{m}{d}'] = np.array(
                    [entry[f'{m}{d}'] for entry in data]
                )

    # 4) Compute VBN if requested
    if 'vbn' in metrics_list:
        kps2D_nn = np.array([entry['kpsL2_NN2D'] for entry in data])
        kps3D_nn = np.array([entry['kpsL2_NN3D'] for entry in data])
        poses = [np.array(entry['pose_t']).reshape((4,4)) for entry in data]
        ts = np.array([p[:3,3] for p in poses])
        dists = np.linalg.norm(ts, axis=1)
        zs = ts[:,2]
        extracted['vbn2D'] = (kps2D_nn * zs) / (fx * dists)
        extracted['vbn3D'] = (kps3D_nn * zs) / (fx * dists)

        # ——— print BDD threshold where 3σ of VBN < 0.01 ———
        if 'bdd' in extracted:
            bdd_vals = extracted['bdd']
            for d in ['2D','3D']:
                vbn = extracted[f'vbn{d}']
                max_bdd = None
                for bb in np.sort(np.unique(bdd_vals)):
                    sigma = np.std(vbn[bdd_vals <= bb])
                    if 3 * sigma < 0.01:
                        max_bdd = bb
                if max_bdd is not None:
                    print(f"Largest BDD where 3σ of VBN{d} < 0.01: {max_bdd:.4f}")
        # ——————————————————————————————————————————————
    # ——— print BDD threshold where mean(IoU) − 3σ > 0.99 ———
    if 'bdd' in extracted and 'iou2D' in extracted:
        bdd_vals = extracted['bdd']
        for d in ['2D','3D']:
            iou_vals = extracted[f'iou{d}']
            max_bdd_iou = None
            for bb in np.sort(np.unique(bdd_vals)):
                sel = iou_vals[bdd_vals <= bb]
                if sel.size == 0:
                    continue
                mu   = np.mean(sel)
                sigma   = np.std(sel)
                if 1 - 3*sigma > 0.9:
                    max_bdd_iou = bb
            if max_bdd_iou is not None:
                print(f"Largest BDD where mean IoU{d} − 3σ > 0.9: {max_bdd_iou:.4f}")
    # ————————————————————————————————————————————————
    # ——— print BDD threshold where mean(IoU) − 3σ > 0.99 ———
    if 'bdd' in extracted and 'ssim2D' in extracted:
        bdd_vals = extracted['bdd']
        for d in ['2D','3D']:
            iou_vals = extracted[f'ssim{d}']
            max_bdd_ssim = None
            for bb in np.sort(np.unique(bdd_vals)):
                sel = iou_vals[bdd_vals <= bb]
                if sel.size == 0:
                    continue
                mu   = np.mean(sel)
                sigma   = np.std(sel)
                if 1 - 3*sigma > 0.9:
                    max_bdd_ssim = bb
            if max_bdd_ssim is not None:
                print(f"Largest BDD where mean SSIM{d} − 3σ > 0.9: {max_bdd_ssim:.4f}")
    # ————————————————————————————————————————————————


    # 5) Setup subplots
    n_rows = len(metrics_list)
    n_cols = len(x_vars) * len(dims)
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(4*n_cols, 4*n_rows),
                            squeeze=False)

    # 6) Plot
    for i, m in enumerate(metrics_list):
        for j, xv in enumerate(x_vars):
            x = extracted[xv]
            
            if xv == "cdist":
                pretty_x = r"C-L2: $\mathbf{||t_1 - t_2||}$ (m)"
            elif xv == "adist":
                pretty_x = r"$\mathrm{\mathbf{S_R}}$: $\mathbf{2\ \mathbf{arccos}(q_r)}$ (rad)"
            elif xv == "comb":
                pretty_x = r"C-L2 + $\mathrm{\mathbf{S_R}}$"
            else:
                pretty_x = xv.upper()

            for k, d in enumerate(dims):
                ax = axs[i, j*len(dims) + k]
                y = extracted[f'{m}{d}']
                color = 'blue' if d=='2D' else 'tab:orange'
                sns.scatterplot(x=x, y=y, ax=ax, color=color, s=1, alpha=0.7)

                # compute correlation
                if show_corr and x.size>1 and np.isfinite(x).all() and np.isfinite(y).all():
                    r = np.corrcoef(x, y)[0,1]
                    r_str = f"r = {r:.2f}"
                else:
                    r_str = ""

                # display labels
                disp = 'Homography T.' if d=='2D' else '3D T.'
                metric_label = '3D Kp. Error / Range' if m=='vbn' else m.upper()

                # Title
                title_main = f"{disp}"
                ax.set_title(title_main,
                              fontsize=15,
                              fontweight='bold',
                              pad=12,
                              loc='left')

                # optional correlation text
                if show_corr and r_str:
                    ax.text(1.0, 1.02, r_str,
                            transform=ax.transAxes,
                            ha='right', va='bottom',
                            fontsize=15, color='red')

                # reference line at 0.01
                ax.axhline(0.1, color='red', linestyle='dotted', linewidth=1.5)

                # Axis labels
                ax.set_xlabel(pretty_x,
                              fontsize=15,
                              fontweight='bold',
                              labelpad=10)
                if k == 0 and i == 0 and j == 0:
                    ax.set_ylabel(f"{metric_label} (%)",
                                fontsize=15,
                                fontweight='bold',
                                labelpad=10)

                # Scales
                if log_x: ax.set_xscale('log')
                if log_y: ax.set_yscale('log')

                # Ticks non-bold, fontsize 10
                ax.tick_params(axis='both',
                               which='major',
                               labelsize=12)
                
                #ax.set_ylim(bottom=-0.05, top=1.05)
    # Set a global title for the whole figure
    #fig.suptitle("SPEED+: Correlation between Distance Metrics and IOU", fontsize=20, fontweight='bold', y=0.9)

    plt.tight_layout(rect=[0,0,1,0.95])
    #fig.set_size_inches(4*n_cols, 4*n_rows)
    plt.show()  
    
if __name__ == "__main__":
    results_file = r"D:\CODE\VISY-REVE-PY\assets\monte_carlo\30000_20250601030136.json"
    plot_mc_results(results_file)

