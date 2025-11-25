#!/usr/bin/env python

import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from mcap_ros2.reader import read_ros2_messages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    TZ_OSLO = ZoneInfo("Europe/Oslo")
except Exception:
    TZ_OSLO = None


def load_rigidbody_data(mcap_path, topic="/rigid_bodies", rb_name="0"):
    """
    Reads /rigid_bodies messages from an MCAP file and returns a dict:
      {
        "t":              (N,)   float64  absolute time [s]
        "frame":          (N,)   int64    frame_number (valid-only)
        "pos":            (N,3) float64  x,y,z
        "quat":           (N,4) float64  [x,y,z,w] (unit quaternions)
        "rpy":            (N,3) float64  roll,pitch,yaw [deg]
        "missing_frames": (M,)   int64   frame_numbers of invalid samples
        "missing_cumsum": (N,)   int64   cumulative missing count at each valid sample index
        "missing_total":  int    total number of invalid samples
        "total_samples":  int    total samples read (valid + invalid)
        "missing_log":    str    path to missing data log file
      }
    """
    times = []
    frames = []
    positions = []
    quats = []

    for msg in read_ros2_messages(mcap_path, topics=[topic]):
        ros_msg = msg.ros_msg

        # Time from header (sec + nanosec)
        t = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec * 1e-9

        # No rigid bodies in this message
        if not ros_msg.rigidbodies:
            continue

        # Select rigid body
        if rb_name is None:
            rb = ros_msg.rigidbodies[0]
        else:
            rb = None
            for rb_candidate in ros_msg.rigidbodies:
                if rb_candidate.rigid_body_name == rb_name:
                    rb = rb_candidate
                    break
            if rb is None:
                # Requested name not present in this message
                continue

        p = rb.pose.position
        q = rb.pose.orientation

        # Build quaternion as numpy array [x, y, z, w]
        q_arr = np.array([q.x, q.y, q.z, q.w], dtype=float)

        times.append(t)
        frames.append(ros_msg.frame_number)
        positions.append([p.x, p.y, p.z])
        quats.append(q_arr)

    if not positions:
        raise RuntimeError(
            "No rigid body data found at all. "
            "Check topic name and rb_name, or that your bag actually contains data."
        )

    times = np.array(times, dtype=float)
    frames = np.array(frames, dtype=int)
    positions = np.array(positions, dtype=float)
    quats = np.array(quats, dtype=float)

    # --- Second-pass filtering of quaternions: remove zeros, NaNs, Infs ---

    norms = np.linalg.norm(quats, axis=1)
    finite_mask = np.isfinite(quats).all(axis=1)
    nonzero_mask = norms > 1e-6
    valid_mask = finite_mask & nonzero_mask

    frames_all = frames.copy()

    n_total = quats.shape[0]
    n_valid = int(valid_mask.sum())
    n_bad = n_total - n_valid

    # Invalid samples in original time/index order
    missing_mask = ~valid_mask
    missing_frames = frames_all[missing_mask]      # frame numbers of invalids
    missing_cumsum_raw = np.cumsum(missing_mask.astype(int))  # length n_total
    missing_cumsum_valid = missing_cumsum_raw[valid_mask]     # aligned with valid samples
    missing_total = int(missing_mask.sum())

    if n_bad > 0:
        print(
            f"[load_rigidbody_data] Filtered out {n_bad} / {n_total} samples "
            f"with invalid quaternions (NaN/Inf or near-zero norm)."
        )
    if n_valid == 0:
        raise RuntimeError(
            "All quaternions were invalid (NaN/Inf or zero norm). "
            "Cannot compute orientations."
        )

    # Apply mask: keep only valid samples for visualization
    times = times[valid_mask]
    frames = frames[valid_mask]
    positions = positions[valid_mask]
    quats = quats[valid_mask]

    # Normalize remaining quaternions
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / norms

    # Convert quaternions to roll/pitch/yaw (deg)
    rot = R.from_quat(quats)  # expects [x,y,z,w]
    rpy = rot.as_euler("xyz", degrees=True)

    # --- Write missing data log file ---

    base, _ = os.path.splitext(mcap_path)
    log_path = base + "_missing_frames.log.txt"

    with open(log_path, "w") as f:
        f.write(f"MCAP file: {mcap_path}\n")
        f.write(f"Topic: {topic}\n")
        f.write(f"Rigid body name: {rb_name}\n\n")
        f.write(f"Total samples read: {n_total}\n")
        f.write(f"Valid samples used: {n_valid}\n")
        f.write(f"Invalid samples (filtered out): {n_bad}\n\n")

        if n_bad > 0:
            f.write("Invalid frames (NaN/Inf/zero-norm quaternions):\n")
            for fr in missing_frames:
                f.write(f"  frame {fr}\n")
        else:
            f.write("No invalid quaternion samples detected.\n")

    print(f"[load_rigidbody_data] Missing data log written to: {log_path}")

    return {
        "t": times,
        "frame": frames,
        "pos": positions,
        "quat": quats,
        "rpy": rpy,
        "missing_frames": missing_frames,
        "missing_cumsum": missing_cumsum_valid,
        "missing_total": missing_total,
        "total_samples": n_total,
        "missing_log": log_path,
    }


def set_equal_aspect_3d(ax, points):
    """
    Set equal aspect ratio for 3D Axes based on given (N,3) points.
    """
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    max_range = (xyz_max - xyz_min).max()

    if max_range == 0:
        max_range = 1.0

    for axis, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        axis(c - max_range / 2.0, c + max_range / 2.0)


def run_visualization(data):
    pos = data["pos"]
    rpy = data["rpy"]
    t = data["t"]
    frames = data["frame"]
    missing_frames = data.get("missing_frames", np.array([], dtype=int))
    missing_cumsum = data.get("missing_cumsum", np.zeros_like(frames, dtype=int))
    n_missing_total = int(data.get("missing_total", missing_frames.size))
    total_samples = int(data.get("total_samples", len(frames) + n_missing_total))

    n = pos.shape[0]

    # Shift time to start at 0 for slider readability
    t0 = t[0]
    t_rel = t - t0

    # Create figure and 3D axes
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    try:
        fig.canvas.manager.set_window_title("Rigid Body 6DoF Visualization")
    except Exception:
        pass

    # ----------------------------------------
    # Scroll-wheel zoom (smooth zoom)
    # ----------------------------------------
    def on_scroll(event):
        # Only react if mouse is over this axes
        if event.inaxes is not ax:
            return

        # Zoom factor: wheel up = zoom in, wheel down = out
        k = 0.9 if event.button == "up" else 1.1

        # Current limits
        x0, x1 = ax.get_xlim3d()
        y0, y1 = ax.get_ylim3d()
        z0, z1 = ax.get_zlim3d()

        # Mouse position in data coords (can be None if outside)
        cx = event.xdata if event.xdata is not None else (x0 + x1) / 2.0
        cy = event.ydata if event.ydata is not None else (y0 + y1) / 2.0
        cz = (z0 + z1) / 2.0  # no zdata for scroll events, use center

        # Scale ranges around (cx, cy, cz)
        ax.set_xlim3d([cx + (x0 - cx) * k, cx + (x1 - cx) * k])
        ax.set_ylim3d([cy + (y0 - cy) * k, cy + (y1 - cy) * k])
        ax.set_zlim3d([cz + (z0 - cz) * k, cz + (z1 - cz) * k])

        fig.canvas.draw_idle()

    # Connect scroll event to handler
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    # Draw ground grid (XY plane at z=0)
    grid_size = max((pos.max(axis=0) - pos.min(axis=0))) * 1.2
    grid_size = max(grid_size, 0.5)
    n_grid = 11
    xs = np.linspace(-grid_size, grid_size, n_grid)
    ys = np.linspace(-grid_size, grid_size, n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    ax.plot_wireframe(X, Y, Z, linewidth=0.5, alpha=0.3)

    # Draw world origin axes (RGB, right-handed)
    axis_len = grid_size * 0.2
    print("grid size:", grid_size, "axis size:", axis_len)
    if axis_len == 0:
        axis_len = 0.2

    # X (red, forward)
    ax.plot([0, axis_len], [0, 0], [0, 0], color="r", linewidth=2)
    # Y (green, right)
    ax.plot([0, 0], [0, axis_len], [0, 0], color="g", linewidth=2)
    # Z (blue, up; we'll invert z-axis so it appears down on screen)
    ax.plot([0, 0], [0, 0], [0, axis_len], color="b", linewidth=2)

    # -------------------------------------------------
    # Custom geometry: boundaries, lines, control room
    # (toggleable via "Bounds" button)
    # -------------------------------------------------

    # 1) Black rectangle boundaries on ground (z=0)
    rect_x = [-8,  8,  8, -8, -8]
    rect_y = [-3, -3,  3,  3, -3]
    rect_z = [ 0,  0,  0,  0,  0]
    boundary_line, = ax.plot(rect_x, rect_y, rect_z, color="k", linewidth=2)

    # 2) Two red lines (at x = -4 and x = 5)
    red_line1, = ax.plot(
        [-4, -4],
        [  3, -3],
        [  0,  0],
        color="r",
        linewidth=2,
    )

    red_line2, = ax.plot(
        [ 5,  5],
        [ 3, -3],
        [ 0,  0],
        color="r",
        linewidth=2,
    )

    # 3) Brown rectangle box "Control Room" (on right short edge)
    # Top rectangle (z = -1) slightly below ground for visual separation
    top_p1 = np.array([ 9,  3, -1])
    top_p2 = np.array([ 9, -3, -1])
    top_p3 = np.array([10, -3, -1])
    top_p4 = np.array([10,  3, -1])

    # Bottom rectangle (z = 0)
    bot_p1 = np.array([ 9,  3, 0])
    bot_p2 = np.array([ 9, -3, 0])
    bot_p3 = np.array([10, -3, 0])
    bot_p4 = np.array([10,  3, 0])

    faces = [
        [top_p1, top_p2, top_p3, top_p4],  # top
        [bot_p1, bot_p2, bot_p3, bot_p4],  # bottom
        [top_p1, top_p2, bot_p2, bot_p1],  # side 1
        [top_p2, top_p3, bot_p3, bot_p2],  # side 2
        [top_p3, top_p4, bot_p4, bot_p3],  # side 3
        [top_p4, top_p1, bot_p1, bot_p4],  # side 4
    ]

    box = Poly3DCollection(
        faces,
        facecolors="#8B4513",
        edgecolors="k",
        alpha=0.8,
    )
    ax.add_collection3d(box)

    # Text "Control Room" at center of top face
    center_top = (top_p1 + top_p2 + top_p3 + top_p4) / 4.0
    control_text = ax.text(
        center_top[0],
        center_top[1],
        center_top[2] + 0.1,
        "Control Room",
        color="k",
        ha="center",
        va="bottom",
        fontsize=10,
        weight="bold",
    )

    # Collect these static points so aspect ratio includes them
    static_points = np.array([
        [-8, -3, 0],
        [ 8, -3, 0],
        [ 8,  3, 0],
        [-8,  3, 0],
        [-4, -3, 0],
        [-4,  3, 0],
        [ 5, -3, 0],
        [ 5,  3, 0],
        top_p1, top_p2, top_p3, top_p4,
        bot_p1, bot_p2, bot_p3, bot_p4,
    ])

    # Main object: ball + body-frame axes
    point = ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], s=50, depthshade=True)

    body_axis_len = 1.0  # Fixed size for moving-object local frame
    body_x_line, = ax.plot([], [], [], color="r", linewidth=2)
    body_y_line, = ax.plot([], [], [], color="g", linewidth=2)
    body_z_line, = ax.plot([], [], [], color="b", linewidth=2)

    # Full trajectory line
    traj_line, = ax.plot([], [], [], "k-", linewidth=1.0, alpha=0.7)
    show_traj = True

    # Partial trajectory (segment) line
    segment_line, = ax.plot([], [], [], "-", linewidth=2.0, alpha=0.9, color="m")
    segment_active = False
    segment_visible = True
    segment_start_idx = None

    # --- Missing-interval markers along trajectory (dynamic, optional) ---
    # We mark positions where missing_cumsum increases (first valid after a gap)
    if n_missing_total > 0 and n > 1:
        jumps = np.zeros(n, dtype=bool)
        jumps[1:] = missing_cumsum[1:] > missing_cumsum[:-1]
        missing_idx = np.where(jumps)[0]
        if missing_idx.size > 0:
            mark_pos = pos[missing_idx]
        else:
            missing_idx = np.array([], dtype=int)
            mark_pos = np.zeros((0, 3))
    else:
        missing_idx = np.array([], dtype=int)
        mark_pos = np.zeros((0, 3))

    # Scatter for markers; initially empty and invisible, updated over time
    missing_markers = ax.scatter([], [], [], marker="x", color="red", s=40, depthshade=False)
    missing_markers.set_visible(False)
    show_missing_marks = False  # state

    # Boundaries visibility state
    show_bounds = True

    # Labels and view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Make Z look "down" visually (inverted axis)
    ax.invert_zaxis()

    # Aspect based on whole path + origin + static geometry
    extra_points = np.vstack([pos, np.zeros((1, 3)), static_points])
    set_equal_aspect_3d(ax, extra_points)

    # Pose text (top-left)
    pose_text = fig.text(
        0.02,
        0.95,
        "",
        fontsize=9,
        va="top",
        family="monospace",
    )

    # Missing data summary (under pose text)
    if n_missing_total == 0:
        missing_str = "Missing data: none (no invalid quaternions)"
    else:
        missing_str = (
            f"Missing data: {n_missing_total} invalid out of {total_samples} total samples"
        )

    missing_text = fig.text(
        0.02,
        0.72,
        missing_str,
        fontsize=8,
        va="top",
        family="monospace",
        color="tab:red" if n_missing_total > 0 else "tab:green",
    )

    # --- Controls: buttons + slider ---

    fig.subplots_adjust(bottom=0.24, left=0.08, right=0.98, top=0.92)

    # Bottom row buttons
    ax_play = fig.add_axes([0.05, 0.03, 0.08, 0.05])
    btn_play = Button(ax_play, "Play")

    ax_traj = fig.add_axes([0.15, 0.03, 0.12, 0.05])
    btn_traj = Button(ax_traj, "Traj: ON")

    ax_seg_start = fig.add_axes([0.29, 0.03, 0.12, 0.05])
    btn_seg_start = Button(ax_seg_start, "Seg Start")

    ax_seg_toggle = fig.add_axes([0.43, 0.03, 0.12, 0.05])
    btn_seg_toggle = Button(ax_seg_toggle, "Seg Show ON")

    ax_seg_reset = fig.add_axes([0.57, 0.03, 0.12, 0.05])
    btn_seg_reset = Button(ax_seg_reset, "Seg Reset")

    # Bounds button
    ax_bounds = fig.add_axes([0.71, 0.03, 0.12, 0.05])
    btn_bounds = Button(ax_bounds, "Bounds ON")

    # Missing markers toggle button
    ax_miss = fig.add_axes([0.85, 0.03, 0.12, 0.05])
    btn_miss = Button(ax_miss, "MissMarks OFF")

    # Time slider (above buttons)
    ax_slider = fig.add_axes([0.15, 0.12, 0.7, 0.03])
    slider = Slider(
        ax_slider,
        "Time [s]",
        valmin=float(t_rel[0]),
        valmax=float(t_rel[-1]),
        valinit=float(t_rel[0]),
    )

    # View buttons (top-right)
    ax_view_top = fig.add_axes([0.80, 0.93, 0.06, 0.04])
    ax_view_side = fig.add_axes([0.87, 0.93, 0.06, 0.04])
    ax_view_bottom = fig.add_axes([0.94, 0.93, 0.06, 0.04])

    btn_view_top = Button(ax_view_top, "Top")
    btn_view_side = Button(ax_view_side, "Side")
    btn_view_bottom = Button(ax_view_bottom, "Bottom")

    # State
    idx = 0
    is_playing = False

    def update_index(new_idx):
        nonlocal idx

        idx = int(np.clip(new_idx, 0, n - 1))

        x, y, z = pos[idx]
        roll, pitch, yaw = rpy[idx]
        time_abs = t[idx]
        time_rel_i = t_rel[idx]
        frame = frames[idx]

        # Human-readable local time
        if TZ_OSLO is not None:
            dt_local = datetime.fromtimestamp(time_abs, TZ_OSLO)
        else:
            dt_local = datetime.fromtimestamp(time_abs)
        human_str = dt_local.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # ms resolution

        # Live missing frame count up to current index (monotonic in time)
        if n_missing_total > 0:
            missing_so_far = int(missing_cumsum[idx])
        else:
            missing_so_far = 0

        # Update ball
        point._offsets3d = ([x], [y], [z])

        # Update body axes based on quaternion -> rotation matrix
        R_body = R.from_quat(data["quat"][idx]).as_matrix()
        origin = np.array([x, y, z])
        axes_body = R_body @ (np.eye(3) * body_axis_len)

        # X axis line (red)
        x_end = origin + axes_body[:, 0]
        body_x_line.set_data([origin[0], x_end[0]], [origin[1], x_end[1]])
        body_x_line.set_3d_properties([origin[2], x_end[2]])

        # Y axis line (green)
        y_end = origin + axes_body[:, 1]
        body_y_line.set_data([origin[0], y_end[0]], [origin[1], y_end[1]])
        body_y_line.set_3d_properties([origin[2], y_end[2]])

        # Z axis line (blue)
        z_end = origin + axes_body[:, 2]
        body_z_line.set_data([origin[0], z_end[0]], [origin[1], z_end[1]])
        body_z_line.set_3d_properties([origin[2], z_end[2]])

        # Update full trajectory
        if show_traj:
            traj_line.set_data(pos[:idx + 1, 0], pos[:idx + 1, 1])
            traj_line.set_3d_properties(pos[:idx + 1, 2])
            traj_line.set_visible(True)
        else:
            traj_line.set_visible(False)

        # Update segment trajectory
        if segment_active and segment_start_idx is not None and idx >= segment_start_idx and segment_visible:
            segment_line.set_data(
                pos[segment_start_idx:idx + 1, 0],
                pos[segment_start_idx:idx + 1, 1],
            )
            segment_line.set_3d_properties(
                pos[segment_start_idx:idx + 1, 2]
            )
            segment_line.set_visible(True)
        else:
            segment_line.set_visible(False)

        # Update missing markers progressively by time/index
        if show_missing_marks and missing_idx.size > 0:
            visible_mask = missing_idx <= idx
            if np.any(visible_mask):
                pts = mark_pos[visible_mask]
                miss_x = pts[:, 0]
                miss_y = pts[:, 1]
                miss_z = np.zeros_like(miss_x)
                missing_markers._offsets3d = (miss_x, miss_y, miss_z)
                missing_markers.set_visible(True)
            else:
                missing_markers._offsets3d = ([], [], [])
                missing_markers.set_visible(False)
        else:
            missing_markers.set_visible(False)

        # Update pose text (top-left) with aligned fields
        L = 18  # adjust width depending on how long your labels are

        line1 = f"{'Time (sec):'.ljust(L)} {time_abs:.6f} s (t0+{time_rel_i:.3f})"
        line2 = f"{'Time (local):'.ljust(L)} {human_str}"
        line3 = f"{'Frame:'.ljust(L)} {frame}"
        line4 = f"{'Position [m]:'.ljust(L)} x={x:+.4f}, y={y:+.4f}, z={z:+.4f}"
        line5 = f"{'Orientation [deg]:'.ljust(L)} roll={roll:+.2f}, pitch={pitch:+.2f}, yaw={yaw:+.2f}"
        line6 = f"{'Missing frames:'.ljust(L)} {missing_so_far}/{n_missing_total}/{total_samples}"

        pose_text.set_text("\n".join([line1, line2, line3, line4, line5, line6]))

        fig.canvas.draw_idle()

    # --- Button / slider callbacks ---

    def on_play_clicked(event):
        nonlocal is_playing
        is_playing = not is_playing
        btn_play.label.set_text("Pause" if is_playing else "Play")

    def on_traj_clicked(event):
        nonlocal show_traj
        show_traj = not show_traj
        btn_traj.label.set_text("Traj: ON" if show_traj else "Traj: OFF")
        update_index(idx)

    def on_seg_start_clicked(event):
        nonlocal segment_active, segment_start_idx, segment_visible
        segment_active = True
        segment_visible = True
        segment_start_idx = idx
        btn_seg_toggle.label.set_text("Seg Show ON")
        update_index(idx)

    def on_seg_toggle_clicked(event):
        nonlocal segment_visible
        segment_visible = not segment_visible
        btn_seg_toggle.label.set_text(
            "Seg Show ON" if segment_visible else "Seg Show OFF"
        )
        update_index(idx)

    def on_seg_reset_clicked(event):
        nonlocal segment_active, segment_visible, segment_start_idx
        segment_active = False
        segment_visible = False
        segment_start_idx = None
        segment_line.set_data([], [])
        segment_line.set_3d_properties([])
        btn_seg_toggle.label.set_text("Seg Show ON")
        fig.canvas.draw_idle()

    def on_bounds_clicked(event):
        nonlocal show_bounds
        show_bounds = not show_bounds
        btn_bounds.label.set_text("Bounds ON" if show_bounds else "Bounds OFF")
        boundary_line.set_visible(show_bounds)
        red_line1.set_visible(show_bounds)
        red_line2.set_visible(show_bounds)
        box.set_visible(show_bounds)
        control_text.set_visible(show_bounds)
        fig.canvas.draw_idle()

    def on_miss_clicked(event):
        nonlocal show_missing_marks
        show_missing_marks = not show_missing_marks
        btn_miss.label.set_text("MissMarks ON" if show_missing_marks else "MissMarks OFF")
        # force refresh to show/hide markers according to current idx
        update_index(idx)

    def on_slider_changed(val):
        # Slider value is relative time
        new_idx = np.searchsorted(t_rel, val)
        update_index(new_idx)

    # View buttons
    def on_view_top(event):
        # Top view: look from -Z
        ax.view_init(elev=-90.0, azim=-90.0)
        fig.canvas.draw_idle()

    def on_view_side(event):
        # Side view: e.g. look along +X
        ax.view_init(elev=180.0, azim=90.0)
        fig.canvas.draw_idle()

    def on_view_bottom(event):
        # Bottom view: look from +Z
        ax.view_init(elev=90.0, azim=-90.0)
        fig.canvas.draw_idle()

    btn_play.on_clicked(on_play_clicked)
    btn_traj.on_clicked(on_traj_clicked)
    btn_seg_start.on_clicked(on_seg_start_clicked)
    btn_seg_toggle.on_clicked(on_seg_toggle_clicked)
    btn_seg_reset.on_clicked(on_seg_reset_clicked)
    btn_bounds.on_clicked(on_bounds_clicked)
    btn_miss.on_clicked(on_miss_clicked)
    slider.on_changed(on_slider_changed)

    btn_view_top.on_clicked(on_view_top)
    btn_view_side.on_clicked(on_view_side)
    btn_view_bottom.on_clicked(on_view_bottom)

    # Timer for animation
    timer = fig.canvas.new_timer(interval=33)  # ~30 FPS

    def on_timer(_):
        if not is_playing:
            return
        if slider.val >= t_rel[-1]:
            # Stop at end
            return
        # Advance time slightly
        dt = (t_rel[-1] - t_rel[0]) / max(n, 1)
        new_val = slider.val + dt
        slider.set_val(new_val)

    timer.add_callback(on_timer, None)
    timer.start()

    # Keyboard shortcut: space toggles play/pause
    def on_key(event):
        if event.key == " ":
            on_play_clicked(None)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Initialize view
    update_index(0)

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 6DoF rigid body motion from MCAP /rigid_bodies"
    )
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to .mcap file",
    )
    parser.add_argument(
        "--topic",
        default="/rigid_bodies",
        help="Topic name (default: /rigid_bodies)",
    )
    parser.add_argument(
        "--rb-name",
        default="0",
        help="Rigid body name to track (string, default: '0'). "
             "Use --rb-name '' to take the first rigid body in each message.",
    )

    args = parser.parse_args()
    rb_name = args.rb_name if args.rb_name != "" else None

    print(f"Loading data from {args.bag} (topic: {args.topic}, rb_name: {rb_name})...")
    data = load_rigidbody_data(args.bag, topic=args.topic, rb_name=rb_name)
    print(f"Loaded {data['pos'].shape[0]} valid samples after filtering.")

    run_visualization(data)


if __name__ == "__main__":
    main()
