import sys
import numpy as np
import cv2
import mapbox_earcut as earcut
import pyray as pr
import tifffile
import topology
import time

def point_dist(p1, p2):
    return (p1.x - p2.x)**2 + (p1.y - p2.y)**2

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_file> [segmentation_file]")
        print("Example: python main.py data/HT-B5.tif optional_seg.npy")
        return
        
    img_path = sys.argv[1]
    seg_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Load Image
    print(f"Loading image {img_path}...")
    if img_path.endswith('.tif') or img_path.endswith('.tiff'):
        img_data = tifffile.imread(img_path)
        if len(img_data.shape) == 3 and img_data.shape[0] in [1, 3, 4]:
            img_data = np.transpose(img_data, (1, 2, 0))
    else:
        img_data = cv2.imread(img_path)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        
    height, width = img_data.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    # Load Segmentation
    if seg_path:
        print(f"Loading segmentation {seg_path}...")
        if seg_path.endswith('.npy'):
            seg_data = np.load(seg_path, allow_pickle=True)
            if seg_data.shape == (): # It's a dict
                seg_data = seg_data.item()['masks']
        else: # bin file
            seg_data = np.fromfile(seg_path, dtype=np.uint16)
            seg_data = seg_data.reshape((height, width))
            
        # Extract Topology
        print("Extracting shared boundary vertices...")
        vertices, regions = topology.extract_boundaries(seg_data)
        print(f"Extracted {len(vertices)} unique shared vertices and {len(regions)} regions.")
    else:
        print("No segmentation map provided. Starting with a blank canvas.")
        vertices = []
        regions = []
    
    # Raylib Initialization
    print("Initialize Window...")
    # Scale down window if image is too large
    scale_factor = 1.0
    max_dim = 1000
    if max(width, height) > max_dim:
        scale_factor = max_dim / max(width, height)
        
    window_w = int(width * scale_factor)
    window_h = int(height * scale_factor)
    
    pr.init_window(window_w, window_h, "Segmentation Editor")
    pr.set_target_fps(60)
    
    # Create background image texture
    # Create raylib image from numpy array
    if len(img_data.shape) == 2: # grayscale
        img_rgb = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
    else:
        # TIF might have alpha or weird channels, assume first 3 are RGB for now
        img_rgb = img_data[:, :, :3]
        if img_rgb.dtype != np.uint8: # normalize to 8-bit if it's 16-bit tiff
             img_rgb = cv2.normalize(img_rgb, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
    # Raylib expects data block, RGBA might be needed depending on system, but pyray usually handles RGB byte arrays manually
    # Easiest way in python w/ pyray is to save to tmp and load
    cv2.imwrite("tmp_bg.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    bg_texture = pr.load_texture("tmp_bg.png")
    
    # State
    dragging_vertex_idx = -1
    hovered_vertex_idx = -1
    last_click_time = 0.0
    selected_region_idx = -1
    selection_time = 0.0
    
    empty_selection_origin = None
    empty_selection_time = 0.0
    
    # Selection radius (scaled by zoom later implicitly by world coordinates)
    PICK_RADIUS = 10.0
    
    # Camera for panning/zooming
    camera = pr.Camera2D()
    camera.target = pr.Vector2(0, 0)
    camera.offset = pr.Vector2(0, 0)
    camera.rotation = 0.0
    camera.zoom = scale_factor

    while not pr.window_should_close():
        
        mouse_pos = pr.get_mouse_position()
        world_mouse_pos = pr.get_screen_to_world_2d(mouse_pos, camera)
        
        # Find hovered vertex globally
        hovered_vertex_idx = -1
        min_dist = float('inf')
        # scale picking radius inversely with zoom so the apparent hit circle size remains constant
        dynamic_pick_radius_sq = (PICK_RADIUS / camera.zoom)**2
        
        # Optimization: in a huge graph, you'd use a generic spatial index here (e.g. quadtree)
        # For ~10k vertices array scan is usually fine in Python for 60fps
        for i, (vx, vy) in enumerate(vertices):
            v_pos = pr.Vector2(vx, vy)
            d = point_dist(world_mouse_pos, v_pos)
            if d < dynamic_pick_radius_sq and d < min_dist:
                min_dist = d
                hovered_vertex_idx = i
        
        # Handle Mouse Input
        if pr.is_mouse_button_pressed(pr.MOUSE_BUTTON_LEFT):
            if hovered_vertex_idx != -1:
                dragging_vertex_idx = hovered_vertex_idx
            else:
                # Check for double click region selection
                current_time = time.time()
                if current_time - last_click_time < 0.5:
                    # Double click detected in empty space, test regions
                    pt = (world_mouse_pos.x, world_mouse_pos.y)
                    clicked_region = -1
                    for r_idx, region in enumerate(regions):
                        poly_pts = [vertices[i] for i in region['vertex_indices']]
                        if len(poly_pts) >= 3:
                            # pointPolygonTest needs float32 numpy array
                            poly_arr = np.array(poly_pts, dtype=np.float32)
                            dist = cv2.pointPolygonTest(poly_arr, pt, False)
                            if dist >= 0:
                                clicked_region = r_idx
                                break
                    if clicked_region != -1:
                        selected_region_idx = clicked_region
                        selection_time = current_time
                        empty_selection_origin = None
                        print(f"Region {selected_region_idx} selected for deletion")
                    else:
                        selected_region_idx = -1
                        empty_selection_origin = pr.Vector2(world_mouse_pos.x, world_mouse_pos.y)
                        empty_selection_time = current_time
                        print(f"Empty space selected for creation at {empty_selection_origin.x}, {empty_selection_origin.y}")
                last_click_time = current_time
                
        elif pr.is_mouse_button_released(pr.MOUSE_BUTTON_LEFT):
            dragging_vertex_idx = -1
                
        # Handle Dragging
        if dragging_vertex_idx != -1:
            vertices[dragging_vertex_idx][0] = world_mouse_pos.x
            vertices[dragging_vertex_idx][1] = world_mouse_pos.y

        # Camera Panning (Right Click)
        if pr.is_mouse_button_down(pr.MOUSE_BUTTON_RIGHT):
            delta = pr.get_mouse_delta()
            delta.x = delta.x * -1.0 / camera.zoom
            delta.y = delta.y * -1.0 / camera.zoom
            camera.target = pr.vector2_add(camera.target, delta)
            
        # Camera Panning (Arrow Keys)
        pan_speed = 10.0 / camera.zoom
        if pr.is_key_down(pr.KEY_RIGHT):
            camera.target.x += pan_speed
        if pr.is_key_down(pr.KEY_LEFT):
            camera.target.x -= pan_speed
        if pr.is_key_down(pr.KEY_DOWN):
            camera.target.y += pan_speed
        if pr.is_key_down(pr.KEY_UP):
            camera.target.y -= pan_speed
            
        # Camera Zooming (Scroll)
        wheel = pr.get_mouse_wheel_move()
        if wheel != 0:
            mouse_world_pos = pr.get_screen_to_world_2d(pr.get_mouse_position(), camera)
            camera.offset = pr.get_mouse_position()
            camera.target = mouse_world_pos
            camera.zoom += wheel * 0.1
            if camera.zoom < 0.1: camera.zoom = 0.1

        # Deletion logic
        if pr.is_key_pressed(pr.KEY_D):
            if selected_region_idx != -1 and time.time() - selection_time < 5.0:
                print(f"Deleting region {selected_region_idx}")
                regions.pop(selected_region_idx)
                selected_region_idx = -1
                
                # Garbage Collect unreferenced vertices to clean up visual clutter
                used_indices = set()
                for r in regions:
                    used_indices.update(r['vertex_indices'])
                    
                # We must rebuild the vertices array to exclude orphans and remap region indices
                new_vertices = []
                index_map = {} # old_idx -> new_idx
                
                for old_idx, v in enumerate(vertices):
                    if old_idx in used_indices:
                        new_idx = len(new_vertices)
                        new_vertices.append(v)
                        index_map[old_idx] = new_idx
                        
                # Reassign vertices mapping
                vertices = new_vertices
                # Remap region references
                for r in regions:
                    r['vertex_indices'] = [index_map[idx] for idx in r['vertex_indices']]
                
                print(f"Garbage collection removed {len(index_map) - len(new_vertices)} orphaned vertices.")

        # Creation Logic
        if pr.is_key_pressed(pr.KEY_N):
            current_time = time.time()
            if empty_selection_origin is not None and current_time - empty_selection_time < 5.0:
                print("Creating new region...")
                # Create 50x50 square centered at empty_selection_origin
                ox, oy = empty_selection_origin.x, empty_selection_origin.y
                half_size = 25.0
                
                new_pts = [
                    [ox - half_size, oy - half_size],
                    [ox + half_size, oy - half_size],
                    [ox + half_size, oy + half_size],
                    [ox - half_size, oy + half_size]
                ]
                
                new_indices = []
                for pt in new_pts:
                    new_indices.append(len(vertices))
                    vertices.append(pt)
                    
                # Find new unique ID
                existing_ids = [r['original_id'] for r in regions]
                new_uid = max(existing_ids) + 1 if existing_ids else 1
                
                color = pr.Color(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255), 255)
                
                regions.append({
                    'original_id': new_uid,
                    'vertex_indices': new_indices,
                    'color': color
                })
                
                empty_selection_origin = None # consume the selection
                print(f"Created region {new_uid}")
            elif selected_region_idx != -1 and current_time - selection_time < 5.0:
                print(f"Adding vertex to region {selected_region_idx}...")
                
                region = regions[selected_region_idx]
                indices = region['vertex_indices']
                
                # Find the closest edge to insert the new vertex
                min_dist = float('inf')
                best_insert_idx = -1
                best_insert_pt = None
                
                m_pt = np.array([world_mouse_pos.x, world_mouse_pos.y])
                
                for i in range(len(indices)):
                    idx1 = indices[i]
                    idx2 = indices[(i+1) % len(indices)]
                    
                    v1 = np.array(vertices[idx1])
                    v2 = np.array(vertices[idx2])
                    
                    # Compute distance from point to line segment
                    l2 = np.sum((v1 - v2)**2)
                    if l2 == 0.0:
                        dist = np.linalg.norm(m_pt - v1)
                        proj_pt = v1
                    else:
                        t = max(0, min(1, np.dot(m_pt - v1, v2 - v1) / l2))
                        proj_pt = v1 + t * (v2 - v1)
                        dist = np.linalg.norm(m_pt - proj_pt)
                        
                    if dist < min_dist:
                        min_dist = dist
                        best_insert_idx = (i + 1) % len(indices)
                        best_insert_pt = proj_pt.tolist()
                        
                if best_insert_idx != -1 and best_insert_pt is not None:
                    # Insert the new vertex into the global array
                    new_v_idx = len(vertices)
                    vertices.append(best_insert_pt)
                    
                    # Insert the reference into the region's topological loop
                    # We insert at `best_insert_idx` to split the edge
                    if best_insert_idx == 0:
                        # if it's the wrap-around edge, append to end
                        indices.append(new_v_idx)
                    else:
                        indices.insert(best_insert_idx, new_v_idx)
                        
                print(f"Added vertex to region {selected_region_idx} at {best_insert_pt}")

        # Saving function
        if pr.is_key_pressed(pr.KEY_S):
            print("Saving modified mask to temporary file...")
            new_mask = topology.reconstruct_mask(vertices, regions, width, height)
            
            # Read original dict to keep image parity
            if seg_path and seg_path.endswith('.npy'):
                orig_data = np.load(seg_path, allow_pickle=True)
                if orig_data.shape == (): # It's a dict
                    new_dict = orig_data.item().copy()
                    new_dict['masks'] = new_mask
                    out_path = "tmp_modified_seg.npy"
                    np.save(out_path, new_dict)
                    print(f"Saved merged dict to {out_path}")
                else:
                    out_path = "tmp_modified_seg.npy"
                    np.save(out_path, new_mask)
                    print(f"Saved array to {out_path}")
            elif seg_path and seg_path.endswith('.bin'):
                out_path = "tmp_modified_seg.bin"
                with open(out_path, "wb") as f:
                    f.write(new_mask.tobytes())
                print(f"Saved binary to {out_path}")
            else:
                out_path = "tmp_modified_seg.npy"
                np.save(out_path, new_mask)
                print(f"Saved brand new array to {out_path}")

        pr.begin_drawing()
        pr.clear_background(pr.RAYWHITE)
        
        pr.begin_mode_2d(camera)
        
        # Draw Background
        pr.draw_texture(bg_texture, 0, 0, pr.WHITE)
        
        # Draw Region Boundaries
        current_time_render = time.time()
        for idx, region in enumerate(regions):
            indices = region['vertex_indices']
            if len(indices) < 2:
                continue
                
            color = region['color']
            
            # Draw filled context if selected for deletion
            if idx == selected_region_idx and (current_time_render - selection_time) < 5.0:
                poly_pts = [vertices[i] for i in indices]
                poly_arr = np.array(poly_pts, dtype=np.float32)
                try:
                    triangles = earcut.triangulate_float32(poly_arr, np.array([len(poly_pts)], dtype=np.uint32))
                    fill_color = pr.Color(color.r, color.g, color.b, 100)
                    for i in range(0, len(triangles), 3):
                        p1 = pr.Vector2(poly_pts[triangles[i]][0], poly_pts[triangles[i]][1])
                        p2 = pr.Vector2(poly_pts[triangles[i+1]][0], poly_pts[triangles[i+1]][1])
                        p3 = pr.Vector2(poly_pts[triangles[i+2]][0], poly_pts[triangles[i+2]][1])
                        # Earcut often generates clockwise, draw backwards
                        pr.draw_triangle(p1, p3, p2, fill_color)
                except Exception:
                    pass
            
            # draw line strip manually
            for i in range(len(indices)):
                idx1 = indices[i]
                idx2 = indices[(i+1) % len(indices)] # wrap around
                v1 = vertices[idx1]
                v2 = vertices[idx2]
                p1 = pr.Vector2(v1[0], v1[1])
                p2 = pr.Vector2(v2[0], v2[1])
                
                # Make lines thicker depending on zoom to be visible
                line_thick = max(1.0, 2.0 / camera.zoom)
                pr.draw_line_ex(p1, p2, line_thick, color)
        
        # Draw Vertices
        # Only draw vertices if zoomed in enough, to prevent clutter on full view
        if camera.zoom > 0.5:
            vert_radius = max(2.0, 3.0 / camera.zoom)
            for i, (vx, vy) in enumerate(vertices):
                color = pr.RED if (i == hovered_vertex_idx or i == dragging_vertex_idx) else pr.BLUE
                pr.draw_circle_v(pr.Vector2(vx, vy), vert_radius, color)
                
        # Draw Creation Crosshair
        if empty_selection_origin is not None and (time.time() - empty_selection_time) < 5.0:
            ch_size = 10.0 / camera.zoom
            ch_thick = max(1.0, 2.0 / camera.zoom)
            p_center = empty_selection_origin
            pr.draw_line_ex(pr.Vector2(p_center.x - ch_size, p_center.y), pr.Vector2(p_center.x + ch_size, p_center.y), ch_thick, pr.RED)
            pr.draw_line_ex(pr.Vector2(p_center.x, p_center.y - ch_size), pr.Vector2(p_center.x, p_center.y + ch_size), ch_thick, pr.RED)

        pr.end_mode_2d()
        
        # UI Overlay
        pr.draw_text("Segmentation Point Editor", 10, 10, 20, pr.BLACK)
        pr.draw_text("Left Click + Drag point: Move boundary", 10, 40, 10, pr.DARKGRAY)
        pr.draw_text("Double Left Click: Select mask/empty space", 10, 55, 10, pr.DARKGRAY)
        pr.draw_text("'D' Key: Delete selected mask | 'N' Key: Create mask", 10, 70, 10, pr.DARKGRAY)
        pr.draw_text("Right Click / Arrows: Pan camera | Mouse Wheel: Zoom", 10, 85, 10, pr.DARKGRAY)
        
        pr.end_drawing()
        
    pr.unload_texture(bg_texture)
    pr.close_window()

if __name__ == "__main__":
    main()
