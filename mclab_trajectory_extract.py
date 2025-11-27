#!/usr/bin/env python

import argparse
import os
import csv
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
        "t":        (N,)   float64   absolute time [s]
        "frame":    (N,)   int64     frame_number (valid-only)
        "pos":      (N,3) float64   x,y,z
        "quat":     (N,4) float64   [x,y,z,w] (unit quaternions)
        "rpy":      (N,3) float64   roll,pitch,yaw [deg]
        "Rmat":     (N,3,3) float64 rotation matrices
        "human_time": list[str]    local time strings
      }
    """
    times = []
    frames = []
    positions = []
    quats = []

    for msg in read_ros2_messages(mcap_path, topics=[topic]):
        ros_msg = msg.ros_msg

        t = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec * 1e-9

        if not ros_msg.rigidbodies:
            continue

        if rb_name is None:
            rb = ros_msg.rigidbodies[0]
        else:
            rb = None
            for rb_candidate in ros_msg.rigidbodies:
                if rb_candidate.rigid_body_name == rb_name:
                    rb = rb_candidate
                    break
            if rb is None:
                continue

        p = rb.pose.position
        q = rb.pose.orientation

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

    # Filter invalid quaternions
    norms = np.linalg.norm(quats, axis=1)
    finite_mask = np.isfinite(quats).all(axis=1)
    nonzero_mask = norms > 1e-6
    valid_mask = finite_mask & nonzero_mask

    n_total = quats.shape[0]
    n_valid = int(valid_mask.sum())
    n_bad = n_total - n_valid

    if n_bad > 0:
        print(
            f"[load_rigidbody_data] Filtered out {n_bad} / {n_total} samples "
            f"with invalid quaternions (NaN/Inf or near-zero norm)."
        )
    if n_valid == 0:
        raise RuntimeError("All quaternions invalid; cannot compute orientations.")

    times = times[valid_mask]
    frames = frames[valid_mask]
    positions = positions[valid_mask]
    quats = quats[valid_mask]

    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / norms

    # Precompute rotation object
    rot = R.from_quat(quats)
    rpy = rot.as_euler("xyz", degrees=True)
    Rmat = rot.as_matrix()  # (N,3,3)

    # Precompute human-readable local times
    human_time = []
    for ti in times:
        if TZ_OSLO is not None:
            dt_local = datetime.fromtimestamp(ti, TZ_OSLO)
        else:
            dt_local = datetime.fromtimestamp(ti)
        human_time.append(dt_local.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

    return {
        "t": times,
        "frame": frames,
        "pos": positions,
        "quat": quats,
        "rpy": rpy,
        "Rmat": Rmat,
        "human_time": human_time,
    }


def set_equal_aspect_3d(ax, points):
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    max_range = (xyz_max - xyz_min).max()
    if max_range == 0:
        max_range = 1.0
    for axis, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        axis(c - max_range / 2.0, c + max_range / 2.0)


def run_visualization(data, bag_path):
    pos = data["pos"]
    rpy = data["rpy"]
    quat = data["quat"]
    t = data["t"]
    frames = data["frame"]
    Rmat_all = data["Rmat"]
    human_time = data["human_time"]

    n = pos.shape[0]
    t0 = t[0]
    t_rel = t - t0

    base, _ = os.path.splitext(bag_path)
    csv_path = base + "_segments.csv"

    # Choose playback step to avoid too many UI updates
    # Control playback step size (skip frames when dataset is large)
    if n <= 20000:
        play_step = 1
    else:
        play_step = max(1, n // 20000)
    print(f"[info] Total samples: {n}, playback step: {play_step}")

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    try:
        fig.canvas.manager.set_window_title(
            "Rigid Body 6DoF Visualization + Segment Export"
        )
    except Exception:
        pass

    # Scroll-wheel zoom
    def on_scroll(event):
        if event.inaxes is not ax:
            return
        k = 0.9 if event.button == "up" else 1.1
        x0, x1 = ax.get_xlim3d()
        y0, y1 = ax.get_ylim3d()
        z0, z1 = ax.get_zlim3d()
        cx = event.xdata if event.xdata is not None else (x0 + x1) / 2.0
        cy = event.ydata if event.ydata is not None else (y0 + y1) / 2.0
        cz = (z0 + z1) / 2.0
        ax.set_xlim3d([cx + (x0 - cx) * k, cx + (x1 - cx) * k])
        ax.set_ylim3d([cy + (y0 - cy) * k, cy + (y1 - cy) * k])
        ax.set_zlim3d([cz + (z0 - cz) * k, cz + (z1 - cz) * k])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    # Ground grid
    grid_size = max((pos.max(axis=0) - pos.min(axis=0))) * 1.2
    grid_size = max(grid_size, 0.5)
    xs = np.linspace(-grid_size, grid_size, 11)
    ys = np.linspace(-grid_size, grid_size, 11)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    ax.plot_wireframe(X, Y, Z, linewidth=0.5, alpha=0.3)

    # World axes
    axis_len = grid_size * 0.2
    if axis_len == 0:
        axis_len = 0.2
    ax.plot([0, axis_len], [0, 0], [0, 0], color="r", linewidth=2)
    ax.plot([0, 0], [0, axis_len], [0, 0], color="g", linewidth=2)
    ax.plot([0, 0], [0, 0], [0, axis_len], color="b", linewidth=2)

    # Boundaries + control room
    rect_x = [-8, 8, 8, -8, -8]
    rect_y = [-3, -3, 3, 3, -3]
    rect_z = [0, 0, 0, 0, 0]
    boundary_line, = ax.plot(rect_x, rect_y, rect_z, color="k", linewidth=2)

    red_line1, = ax.plot([-4, -4], [3, -3], [0, 0], color="r", linewidth=2)
    red_line2, = ax.plot([5, 5], [3, -3], [0, 0], color="r", linewidth=2)

    top_p1 = np.array([9, 3, -1])
    top_p2 = np.array([9, -3, -1])
    top_p3 = np.array([10, -3, -1])
    top_p4 = np.array([10, 3, -1])
    bot_p1 = np.array([9, 3, 0])
    bot_p2 = np.array([9, -3, 0])
    bot_p3 = np.array([10, -3, 0])
    bot_p4 = np.array([10, 3, 0])

    faces = [
        [top_p1, top_p2, top_p3, top_p4],
        [bot_p1, bot_p2, bot_p3, bot_p4],
        [top_p1, top_p2, bot_p2, bot_p1],
        [top_p2, top_p3, bot_p3, bot_p2],
        [top_p3, top_p4, bot_p4, bot_p3],
        [top_p4, top_p1, bot_p1, bot_p4],
    ]
    box = Poly3DCollection(faces, facecolors="#8B4513", edgecolors="k", alpha=0.8)
    ax.add_collection3d(box)

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

    static_points = np.array([
        [-8, -3, 0],
        [8, -3, 0],
        [8, 3, 0],
        [-8, 3, 0],
        [-4, -3, 0],
        [-4, 3, 0],
        [5, -3, 0],
        [5, 3, 0],
        top_p1, top_p2, top_p3, top_p4,
        bot_p1, bot_p2, bot_p3, bot_p4,
    ])

    base_point_color = "C0"
    point = ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2],
                       s=50, depthshade=True, c=base_point_color)

    body_axis_len = 1.0
    body_x_line, = ax.plot([], [], [], color="r", linewidth=2)
    body_y_line, = ax.plot([], [], [], color="g", linewidth=2)
    body_z_line, = ax.plot([], [], [], color="b", linewidth=2)

    traj_line, = ax.plot([], [], [], "k-", linewidth=1.0, alpha=0.7)
    show_traj = True

    rec_traj_line, = ax.plot([], [], [], "-", linewidth=2.0, alpha=0.9, color="orange")
    show_rec_traj = True

    show_bounds = True

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.invert_zaxis()

    extra_points = np.vstack([pos, np.zeros((1, 3)), static_points])
    set_equal_aspect_3d(ax, extra_points)

    pose_text = fig.text(
        0.02, 0.95, "", fontsize=9, va="top", family="monospace"
    )

    rec_indicator = fig.text(
        0.88, 0.90, "REC ●", fontsize=12, va="top", ha="left",
        color="gray", family="monospace", weight="bold"
    )

    # Recording state
    recording = False
    record_mask = np.zeros(n, dtype=bool)
    traj_id = np.zeros(n, dtype=int)
    current_traj_id = 0

    fig.subplots_adjust(bottom=0.24, left=0.08, right=0.98, top=0.92)

    ax_play = fig.add_axes([0.03, 0.03, 0.08, 0.05])
    btn_play = Button(ax_play, "Play")

    ax_traj_btn = fig.add_axes([0.13, 0.03, 0.12, 0.05])
    btn_traj = Button(ax_traj_btn, "Traj: ON")

    ax_rec = fig.add_axes([0.27, 0.03, 0.12, 0.05])
    btn_rec = Button(ax_rec, "Rec OFF")
    rec_off_color = btn_rec.color
    rec_off_hover = btn_rec.hovercolor
    rec_on_color = "#ffaaaa"
    rec_on_hover = "#ff8888"

    ax_rec_traj_btn = fig.add_axes([0.41, 0.03, 0.14, 0.05])
    btn_rec_traj = Button(ax_rec_traj_btn, "RecTraj ON")

    ax_reset = fig.add_axes([0.57, 0.03, 0.12, 0.05])
    btn_reset = Button(ax_reset, "ResetRec")

    ax_save = fig.add_axes([0.71, 0.03, 0.12, 0.05])
    btn_save = Button(ax_save, "Save CSV")

    ax_bounds_btn = fig.add_axes([0.85, 0.03, 0.12, 0.05])
    btn_bounds = Button(ax_bounds_btn, "Bounds ON")

    ax_slider = fig.add_axes([0.15, 0.12, 0.7, 0.03])
    slider = Slider(
        ax_slider, "Time [s]",
        valmin=float(t_rel[0]),
        valmax=float(t_rel[-1]),
        valinit=float(t_rel[0]),
    )

    ax_view_top = fig.add_axes([0.80, 0.93, 0.06, 0.04])
    ax_view_side = fig.add_axes([0.87, 0.93, 0.06, 0.04])
    ax_view_bottom = fig.add_axes([0.94, 0.93, 0.06, 0.04])
    btn_view_top = Button(ax_view_top, "Top")
    btn_view_side = Button(ax_view_side, "Side")
    btn_view_bottom = Button(ax_view_bottom, "Bottom")

    idx = 0
    is_playing = False

    def update_index(new_idx, allow_record=True):
        """
        Update visualization to a new index.

        If recording is ON and allow_record=True, mark the entire
        index range between the previous idx and new_idx as recorded,
        and assign the current trajectory ID to that range.
        """
        nonlocal idx, record_mask, traj_id

        new_idx = int(np.clip(new_idx, 0, n - 1))

        if recording and allow_record and current_traj_id > 0:
            i0 = min(idx, new_idx)
            i1 = max(idx, new_idx)
            record_mask[i0:i1 + 1] = True
            traj_id[i0:i1 + 1] = current_traj_id

        idx = new_idx

        x, y, z = pos[idx]
        roll, pitch, yaw = rpy[idx]
        time_abs = t[idx]
        time_rel_i = t_rel[idx]
        frame = frames[idx]
        human_str = human_time[idx]
        R_body = Rmat_all[idx]

        if recording:
            point.set_color("red")
        else:
            point.set_color(base_point_color)

        point._offsets3d = ([x], [y], [z])

        origin = np.array([x, y, z])
        axes_body = R_body @ (np.eye(3) * body_axis_len)

        x_end = origin + axes_body[:, 0]
        body_x_line.set_data([origin[0], x_end[0]], [origin[1], x_end[1]])
        body_x_line.set_3d_properties([origin[2], x_end[2]])

        y_end = origin + axes_body[:, 1]
        body_y_line.set_data([origin[0], y_end[0]], [origin[1], y_end[1]])
        body_y_line.set_3d_properties([origin[2], y_end[2]])

        z_end = origin + axes_body[:, 2]
        body_z_line.set_data([origin[0], z_end[0]], [origin[1], z_end[1]])
        body_z_line.set_3d_properties([origin[2], z_end[2]])

        if show_traj:
            traj_line.set_data(pos[:idx + 1, 0], pos[:idx + 1, 1])
            traj_line.set_3d_properties(pos[:idx + 1, 2])
            traj_line.set_visible(True)
        else:
            traj_line.set_visible(False)

        if show_rec_traj:
            m = idx + 1
            x_rec = pos[:m, 0].copy()
            y_rec = pos[:m, 1].copy()
            z_rec = pos[:m, 2].copy()
            mask_not_rec = ~record_mask[:m]
            x_rec[mask_not_rec] = np.nan
            y_rec[mask_not_rec] = np.nan
            z_rec[mask_not_rec] = np.nan
            rec_traj_line.set_data(x_rec, y_rec)
            rec_traj_line.set_3d_properties(z_rec)
            rec_traj_line.set_visible(True)
        else:
            rec_traj_line.set_visible(False)

        n_rec = int(record_mask.sum())
        traj_here = int(traj_id[idx])

        L = 18
        line1 = f"{'Time (sec):'.ljust(L)} {time_abs:.6f} s (t0+{time_rel_i:.3f})"
        line2 = f"{'Time (local):'.ljust(L)} {human_str}"
        line3 = f"{'Frame:'.ljust(L)} {frame}"
        line4 = f"{'Position [m]:'.ljust(L)} x={x:+.4f}, y={y:+.4f}, z={z:+.4f}"
        line5 = f"{'Orientation [deg]:'.ljust(L)} roll={roll:+.2f}, pitch={pitch:+.2f}, yaw={yaw:+.2f}"
        line6 = f"{'Recording:'.ljust(L)} {'ON' if recording else 'OFF'}"
        line7 = f"{'Recorded samples:'.ljust(L)} {n_rec}"
        line8 = f"{'Trajectory no:'.ljust(L)} {traj_here} (total={current_traj_id})"
        line9 = f"{'Play step:'.ljust(L)} {play_step}"

        pose_text.set_text("\n".join(
            [line1, line2, line3, line4, line5, line6, line7, line8, line9]
        ))

        rec_indicator.set_color("red" if recording else "gray")

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
        update_index(idx, allow_record=False)

    def on_rec_clicked(event):
        nonlocal recording, current_traj_id, record_mask, traj_id

        old_state = recording
        new_state = not recording

        # When turning OFF: clamp any future samples with this traj ID
        if old_state and not new_state and current_traj_id > 0:
            future_idx = np.where((traj_id == current_traj_id) & (np.arange(n) > idx))[0]
            if future_idx.size > 0:
                record_mask[future_idx] = False
                traj_id[future_idx] = 0

        recording = new_state

        if (not old_state) and recording:
            current_traj_id += 1

        btn_rec.label.set_text("Rec ON" if recording else "Rec OFF")
        if recording:
            btn_rec.color = rec_on_color
            btn_rec.hovercolor = rec_on_hover
        else:
            btn_rec.color = rec_off_color
            btn_rec.hovercolor = rec_off_hover

        update_index(idx, allow_record=False)

    def on_rec_traj_clicked(event):
        nonlocal show_rec_traj
        show_rec_traj = not show_rec_traj
        btn_rec_traj.label.set_text("RecTraj ON" if show_rec_traj else "RecTraj OFF")
        update_index(idx, allow_record=False)

    def on_reset_clicked(event):
        nonlocal recording, record_mask, traj_id, current_traj_id
        recording = False
        btn_rec.label.set_text("Rec OFF")
        btn_rec.color = rec_off_color
        btn_rec.hovercolor = rec_off_hover
        rec_indicator.set_color("gray")

        record_mask[:] = False
        traj_id[:] = 0
        current_traj_id = 0

        rec_traj_line.set_data([], [])
        rec_traj_line.set_3d_properties([])

        update_index(idx, allow_record=False)

    def on_save_clicked(event):
        selected_idx = np.nonzero(record_mask)[0]
        if selected_idx.size == 0:
            print("[export] No samples recorded. CSV not written.")
            return
        selected_idx = np.sort(selected_idx)

        header = [
            "Time_sec",
            "Time_local",
            "Frame",
            "Trajectory_no",
            "Pos_x",
            "Pos_y",
            "Pos_z",
            "Roll_deg",
            "Pitch_deg",
            "Yaw_deg",
            "Quat_x",
            "Quat_y",
            "Quat_z",
            "Quat_w",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(header)
            for i in selected_idx:
                time_abs = t[i]
                time_local_str = human_time[i]
                x, y, z = pos[i]
                roll, pitch, yaw = rpy[i]
                qx, qy, qz, qw = quat[i]
                frame_i = int(frames[i])
                traj_i = int(traj_id[i])

                writer.writerow([
                    f"{time_abs:.9f}",
                    time_local_str,
                    frame_i,
                    traj_i,
                    f"{x:.9f}",
                    f"{y:.9f}",
                    f"{z:.9f}",
                    f"{roll:.6f}",
                    f"{pitch:.6f}",
                    f"{yaw:.6f}",
                    f"{qx:.9f}",
                    f"{qy:.9f}",
                    f"{qz:.9f}",
                    f"{qw:.9f}",
                ])

        print(f"[export] Saved {selected_idx.size} samples to: {csv_path}")

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

    def on_slider_changed(val):
        new_idx = np.searchsorted(t_rel, val)
        update_index(new_idx)

    def on_view_top(event):
        ax.view_init(elev=-90.0, azim=-90.0)
        fig.canvas.draw_idle()

    def on_view_side(event):
        ax.view_init(elev=180.0, azim=90.0)
        fig.canvas.draw_idle()

    def on_view_bottom(event):
        ax.view_init(elev=90.0, azim=-90.0)
        fig.canvas.draw_idle()

    btn_play.on_clicked(on_play_clicked)
    btn_traj.on_clicked(on_traj_clicked)
    btn_rec.on_clicked(on_rec_clicked)
    btn_rec_traj.on_clicked(on_rec_traj_clicked)
    btn_reset.on_clicked(on_reset_clicked)
    btn_save.on_clicked(on_save_clicked)
    btn_bounds.on_clicked(on_bounds_clicked)
    slider.on_changed(on_slider_changed)

    btn_view_top.on_clicked(on_view_top)
    btn_view_side.on_clicked(on_view_side)
    btn_view_bottom.on_clicked(on_view_bottom)

    # Slightly slower timer to reduce CPU load
    timer = fig.canvas.new_timer(interval=40)  # ~25 FPS target

    def on_timer(_):
        if not is_playing:
            return
        if slider.val >= t_rel[-1]:
            return
        dt = play_step * (t_rel[-1] - t_rel[0]) / max(n, 1)
        slider.set_val(slider.val + dt)

    timer.add_callback(on_timer, None)
    timer.start()

    def on_key(event):
        if event.key == " ":
            on_play_clicked(None)

    fig.canvas.mpl_connect("key_press_event", on_key)

    update_index(0, allow_record=False)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 6DoF rigid body motion from MCAP and export selected segments to CSV"
    )
    parser.add_argument("--bag", required=True, help="Path to .mcap file")
    parser.add_argument("--topic", default="/rigid_bodies", help="Topic name")
    parser.add_argument(
        "--rb-name",
        default="0",
        help="Rigid body name (default: '0', use '' for first in message)",
    )

    args = parser.parse_args()
    rb_name = args.rb_name if args.rb_name != "" else None

    print(f"Loading data from {args.bag} (topic: {args.topic}, rb_name: {rb_name})...")
    data = load_rigidbody_data(args.bag, topic=args.topic, rb_name=rb_name)
    print(f"Loaded {data['pos'].shape[0]} valid samples after filtering.")

    run_visualization(data, bag_path=args.bag)


if __name__ == "__main__":
    main()
