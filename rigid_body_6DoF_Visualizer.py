#!/usr/bin/env python

import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from mcap_ros2.reader import read_ros2_messages


def load_rigidbody_data(mcap_path, topic="/rigid_bodies", rb_name="0"):
    """
    Reads /rigid_bodies messages from an MCAP file and returns a dict:
      {
        "t":              (N,)   float64  absolute time [s]
        "frame":          (N,)   int64    frame_number
        "pos":            (N,3) float64  x,y,z
        "quat":           (N,4) float64  [x,y,z,w] (unit quaternions)
        "rpy":            (N,3) float64  roll,pitch,yaw [deg]
        "missing_frames": (M,)   int64    frames filtered out due to invalid quats
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

    missing_frames = frames_all[~valid_mask]

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

    # Apply mask
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
    if axis_len == 0:
        axis_len = 0.2

    # X (red, forward)
    ax.plot([0, axis_len], [0, 0], [0, 0], color="r", linewidth=2)
    # Y (green, right)
    ax.plot([0, 0], [0, axis_len], [0, 0], color="g", linewidth=2)
    # Z (blue, up; we'll invert z-axis so it appears down on screen)
    ax.plot([0, 0], [0, 0], [0, axis_len], color="b", linewidth=2)

    # Main object: ball + body-frame axes
    # Ball
    point = ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], s=50, depthshade=True)

    # Body axes (3 lines for x,y,z)
    body_axis_len = axis_len * 0.7
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

    # Labels and view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Make Z look "down" visually (inverted axis)
    ax.invert_zaxis()

    # Aspect based on whole path + origin
    extra_points = np.vstack([pos, np.zeros((1, 3))])
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

    # Missing data text (under pose text)
    if missing_frames.size == 0:
        missing_str = "Missing data: none (no invalid quaternions)"
    else:
        first_list = ", ".join(str(int(f)) for f in missing_frames[:10])
        more = "" if missing_frames.size <= 10 else f", ... (+{missing_frames.size - 10} more)"
        missing_str = (
            f"Missing data: {missing_frames.size} out of {frames.size + missing_frames.size} frames"
        )

    missing_text = fig.text(
        0.02,
        0.72,
        missing_str,
        fontsize=8,
        va="top",
        family="monospace",
        color="tab:red" if missing_frames.size > 0 else "tab:green",
    )

    # --- Controls: buttons + slider ---

    # Shrink 3D axes to leave room for controls at bottom
    fig.subplots_adjust(bottom=0.24, left=0.08, right=0.98, top=0.92)

    # Bottom row buttons
    ax_play = fig.add_axes([0.08, 0.03, 0.10, 0.05])
    btn_play = Button(ax_play, "Play")

    ax_traj = fig.add_axes([0.20, 0.03, 0.14, 0.05])
    btn_traj = Button(ax_traj, "Traj: ON")

    ax_seg_start = fig.add_axes([0.38, 0.03, 0.14, 0.05])
    btn_seg_start = Button(ax_seg_start, "Seg Start")

    ax_seg_toggle = fig.add_axes([0.56, 0.03, 0.14, 0.05])
    btn_seg_toggle = Button(ax_seg_toggle, "Seg Show ON")

    ax_seg_reset = fig.add_axes([0.74, 0.03, 0.14, 0.05])
    btn_seg_reset = Button(ax_seg_reset, "Seg Reset")

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
        time_rel = t_rel[idx]
        frame = frames[idx]

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

        # Update pose text (top-left)
        L = 18  # adjust width depending on how long your labels are

        line1 = f"{'Time:'.ljust(L)} {time_abs:.6f} s (t0+{time_rel:.3f})"
        line2 = f"{'Frame:'.ljust(L)} {frame}"
        line3 = f"{'Position [m]:'.ljust(L)} x={x:+.4f}, y={y:+.4f}, z={z:+.4f}"
        line4 = f"{'Orientation [deg]:'.ljust(L)} roll={roll:+.2f}, pitch={pitch:+.2f}, yaw={yaw:+.2f}"

        pose_text.set_text("\n".join([line1, line2, line3, line4]))


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

    def on_slider_changed(val):
        # Slider value is relative time
        new_idx = np.searchsorted(t_rel, val)
        update_index(new_idx)

    # View buttons
    def on_view_top(event):
        # Top view: look from +Z down
        ax.view_init(elev=90.0, azim=-90.0)
        fig.canvas.draw_idle()

    def on_view_side(event):
        # Side view: e.g. look along +X
        ax.view_init(elev=180.0, azim=90.0)
        fig.canvas.draw_idle()

    def on_view_bottom(event):
        # Bottom view: look from -Z up
        ax.view_init(elev=-90.0, azim=-90.0)
        fig.canvas.draw_idle()

    btn_play.on_clicked(on_play_clicked)
    btn_traj.on_clicked(on_traj_clicked)
    btn_seg_start.on_clicked(on_seg_start_clicked)
    btn_seg_toggle.on_clicked(on_seg_toggle_clicked)
    btn_seg_reset.on_clicked(on_seg_reset_clicked)
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

