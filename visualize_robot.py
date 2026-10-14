#!/usr/bin/env python3
"""
Full robot visualization using a URDF with meshes.
"""

import argparse
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import time
import os

def main():
    parser = argparse.ArgumentParser(description="Full robot visualization")
    parser.add_argument('--log', required=True, help="Log file path (.npz)")
    parser.add_argument('--urdf', required=True, help="Robot URDF file path")
    parser.add_argument('--mesh-search-path', help="Directory containing URDF mesh packages")
    parser.add_argument('--fps', type=int, default=1000, help="Playback FPS (default: 50)")
    parser.add_argument('--loop', action='store_true', help="Loop playback")
    parser.add_argument('--start', type=int, default=0, help="Start frame index")
    parser.add_argument('--end', type=int, default=-1, help="End frame index")
    parser.add_argument('--track', action='store_true', help="Camera tracks the robot base")
    args = parser.parse_args()
    
    urdf_path = args.urdf
    mesh_search_path = args.mesh_search_path or os.path.dirname(urdf_path)
    # Load data
    print(f"Loading log: {args.log}")
    log_path = args.log
    data = np.load(log_path, allow_pickle=True)
    t = data['t']
    q = data['q']
    
    n_samples = len(t)
    start_idx = max(0, args.start)
    end_idx = n_samples if args.end == -1 else min(args.end, n_samples)
    
    print(f"  Samples: {n_samples}, playback frames: {start_idx}-{end_idx}")
    
    # ✅ Check joint names
    joint_names_data = None
    if 'joint_names' in data:
        joint_names_data = [str(name) for name in data['joint_names']]
        print(f"  ✅ Data contains joint name information ({len(joint_names_data)} joints)")
    else:
        print(f"  ⚠️  Data is missing joint name information")
    
    try:
        # Load model
        print(f"Loading URDF...")
        model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        data_pin = model.createData()
        print(f"  ✓ Model loaded (nq={model.nq})")
        
        # Load geometry
        print(f"Loading geometry...")
        visual_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL, mesh_search_path)
        collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION, mesh_search_path)
        
        print(f"  ✓ Geometry loaded successfully!")
        print(f"    - visual: {visual_model.ngeoms}")
        print(f"    - collision: {collision_model.ngeoms}")
        has_geometry = True
        
    except Exception as e:
        print(f"  ⚠️  Failed to load: {e}")
        print(f"  Using simplified visualization...")
        visual_model = pin.GeometryModel()
        collision_model = pin.GeometryModel()
        has_geometry = False
    
    
    # ✅ Extract joint order from Pinocchio model
    joint_names_pinocchio = []
    for i in range(2, model.njoints):  # Skip universe and free-flyer
        joint_names_pinocchio.append(model.names[i])
    
    print(f"\nPinocchio joint order ({len(joint_names_pinocchio)} joints):")
    for i, name in enumerate(joint_names_pinocchio[:5]):
        print(f"  [{i:2d}] {name}")
    if len(joint_names_pinocchio) > 5:
        print(f"  ... ({len(joint_names_pinocchio) - 5} more)")
    
    # ✅ Check whether joint orders match
    needs_reorder = False
    if joint_names_data is not None:
        if joint_names_data == joint_names_pinocchio:
            print("\n✅ Joint order matches, no reordering needed")
            needs_reorder = False
        else:
            print("\n⚠️  Joint order mismatch, reordering data...")
            needs_reorder = True
            
            # Build mapping: for each Pinocchio joint, find its index in the data
            # reorder_indices[pinocchio_idx] = data_idx
            joint_name_to_data_idx = {name: i for i, name in enumerate(joint_names_data)}
            reorder_indices = []
            for pinocchio_name in joint_names_pinocchio:
                if pinocchio_name not in joint_name_to_data_idx:
                    raise ValueError(f"Pinocchio joint '{pinocchio_name}' not found in data!")
                reorder_indices.append(joint_name_to_data_idx[pinocchio_name])
            reorder_indices = np.array(reorder_indices)
            
            # Reorder joint data for all frames
            print("Reordering joint data...")
            print(f"  Example: Pinocchio[0] ({joint_names_pinocchio[0]}) <- "
                  f"Data[{reorder_indices[0]}] ({joint_names_data[reorder_indices[0]]})")
            q_reordered = q.copy()
            for i in range(n_samples):
                # q[i] = [base_pos(3), base_quat(4), joints(29)]
                q_reordered[i, 7:] = q[i, 7:][reorder_indices]
            q = q_reordered
            print("✅ Joint data reordering completed")
    else:
        print("\n⚠️  Assuming joint order matches (cannot verify)")
    
    # ✅ Convert quaternion format: the log stores [w,x,y,z], Pinocchio expects [x,y,z,w]
    print(f"\nConverting quaternion format...")
    print(f"  Data format: [pos(3), quat_wxyz(4), joints({len(joint_names_pinocchio)})]")
    print(f"  Pinocchio: [pos(3), quat_xyzw(4), joints({len(joint_names_pinocchio)})]")
    
    q_converted = q.copy()
    for i in range(n_samples):
        # q[i] = [x, y, z, qw, qx, qy, qz, joint_pos...]
        # Need to convert to [x, y, z, qx, qy, qz, qw, joint_pos...]
        qw, qx, qy, qz = q[i, 3], q[i, 4], q[i, 5], q[i, 6]
        q_converted[i, 3] = qx  # Pinocchio index 3 = qx
        q_converted[i, 4] = qy  # Pinocchio index 4 = qy
        q_converted[i, 5] = qz  # Pinocchio index 5 = qz
        q_converted[i, 6] = qw  # Pinocchio index 6 = qw
    
    q = q_converted
    print(f"  ✅ Quaternion format conversion completed")
    
    # Check dimensions
    if q.shape[1] != model.nq:
        print(f"\n⚠️  Data dimension: {q.shape[1]} vs {model.nq}")
        if q.shape[1] < model.nq:
            q_full = np.zeros((n_samples, model.nq))
            q_full[:, 3] = 1.0
            q_full[:, 7:7+q.shape[1]] = q
            q = q_full
            print(f"  Padded to: {q.shape}")
    
    # Initialize Meshcat
    print("\nInitializing Meshcat...")
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    
    if has_geometry:
        try:
            viz.loadViewerModel()
            print("✓ Full robot model loaded in browser!")
        except Exception as e:
            print(f"⚠️  Failed to load model into viewer: {e}")
            has_geometry = False
    
    if not has_geometry:
        print("✓ Simplified visualization started")
    
    print(f"\nOpen the displayed URL in your browser")
    print(f"You should see {'the full G1 robot' if has_geometry else 'coordinate axes'}!\n")
    
    # Display initial pose
    q_init = q[start_idx]
    pin.forwardKinematics(model, data_pin, q_init)
    pin.updateFramePlacements(model, data_pin)
    
    try:
        viz.display(q_init)
    except:
        # Fall back silently if display fails
        pass
    
    print(f"Starting playback (FPS: {args.fps})...")
    print("Press Ctrl+C to stop\n")
    
    # Compute time interval per frame
    dt = 1.0 / args.fps
    
    loop_count = 0
    try:
        while True:
            loop_count += 1
            if args.loop and loop_count > 1:
                print(f"\nLoop #{loop_count}")
            
            for i in range(start_idx, end_idx):
                start_time = time.time()
                q_i = q[i]
                
                # Update state
                pin.forwardKinematics(model, data_pin, q_i)
                pin.updateFramePlacements(model, data_pin)
                
                # Camera tracking
                if args.track:
                    try:
                        base_pos = q_i[:3]
                        # Keep camera at a fixed offset (e.g. 3m back, 1.5m up)
                        # Note: Meshcat uses Z-up
                        viz.viewer.set_cam_target(base_pos)
                        viz.viewer.set_cam_pos(base_pos + np.array([-3.0, 0.0, 1.5]))
                    except Exception:
                        pass

                # Render
                try:
                    viz.display(q_i)
                except:
                    pass
                
                # Progress
                if i % 50 == 0 or i == end_idx - 1:
                    progress = (i - start_idx + 1) / (end_idx - start_idx) * 100
                    base_pos = q_i[:3]
                    print(f"\rStep: {i}/{end_idx} | Progress: {progress:5.1f}% | "
                          f"t={t[i]:6.2f}s | base_z={base_pos[2]:.2f}m",
                          end='', flush=True)
                
                # Control frame rate
                elapsed = time.time() - start_time
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            print()
            
            if not args.loop:
                break
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    
    print("\nPlayback finished. Window stays open; press Ctrl+C to exit.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExit")


if __name__ == "__main__":
    main()
