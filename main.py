import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from scipy.interpolate import CubicSpline
from functools import partial
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'morphology'))

MORPHOLOGY_CPP_AVAILABLE = False
morphology_cpp = None
_morphology_error = None

try:
    import morphology_cpp
    MORPHOLOGY_CPP_AVAILABLE = True
except (ImportError, ValueError, RuntimeError) as e:
    _morphology_error = f"{type(e).__name__}: {e}"
    print(f"Error: C++ morphology module not available. Morphology operations will not work.")
    print(f"Details: {_morphology_error}")


class GammaCurveEditor(tk.Toplevel):
    def __init__(self, parent, callback, initial_lut=None):
        super().__init__(parent)
        self.title("Gamma Curve Editor")
        self.callback = callback
        
        self.current_channel = 'R'
        
        self.control_points = {
            'R': [(0, 0), (255, 255)],
            'G': [(0, 0), (255, 255)],
            'B': [(0, 0), (255, 255)]
        }
        
        self.lut = {
            'R': np.arange(256, dtype=np.uint8),
            'G': np.arange(256, dtype=np.uint8),
            'B': np.arange(256, dtype=np.uint8)
        }
        
        if initial_lut is not None:
            self.lut = initial_lut.copy()
        
        self.dragging_point = None
        self.point_radius = 6
        
        self._create_ui()
        self._update_lut()
        
    def _create_ui(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        channel_frame = ttk.LabelFrame(main_frame, text="Channel Selection", padding=5)
        channel_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.channel_var = tk.StringVar(value='R')
        ttk.Radiobutton(channel_frame, text="Red (R)", variable=self.channel_var, value='R', command=self._on_channel_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(channel_frame, text="Green (G)", variable=self.channel_var, value='G', command=self._on_channel_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(channel_frame, text="Blue (B)", variable=self.channel_var, value='B', command=self._on_channel_change).pack(side=tk.LEFT, padx=5)
        
        canvas_frame = ttk.LabelFrame(main_frame, text="Gamma Curve (Right-click to add point)", padding=5)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_size = 300
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size, bg='white', highlightthickness=1)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Reset Current", command=self._reset_current).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset All", command=self._reset_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Apply", command=self._apply).pack(side=tk.RIGHT, padx=5)
        
        self._draw_curve()
        
    def _on_channel_change(self):
        self.current_channel = self.channel_var.get()
        self._draw_curve()
        
    def _get_canvas_coords(self, x, y):
        margin = 20
        plot_size = self.canvas_size - 2 * margin
        cx = margin + (x / 255.0) * plot_size
        cy = margin + (1 - y / 255.0) * plot_size
        return cx, cy
        
    def _get_image_coords(self, cx, cy):
        margin = 20
        plot_size = self.canvas_size - 2 * margin
        x = max(0, min(255, int(round((cx - margin) / plot_size * 255))))
        y = max(0, min(255, int(round((1 - (cy - margin) / plot_size) * 255))))
        return x, y
        
    def _find_point(self, cx, cy):
        points = self.control_points[self.current_channel]
        for i, (x, y) in enumerate(points):
            px, py = self._get_canvas_coords(x, y)
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist < self.point_radius + 5:
                return i
        return None
        
    def _on_left_click(self, event):
        idx = self._find_point(event.x, event.y)
        if idx is not None and idx > 0 and idx < len(self.control_points[self.current_channel]) - 1:
            self.dragging_point = idx
            
    def _on_drag(self, event):
        if self.dragging_point is not None:
            points = self.control_points[self.current_channel]
            x, y = self._get_image_coords(event.x, event.y)
            
            prev_idx = self.dragging_point - 1
            next_idx = self.dragging_point + 1
            
            if prev_idx >= 0:
                x = max(points[prev_idx][0] + 1, x)
            if next_idx < len(points):
                x = min(points[next_idx][0] - 1, x)
                
            points[self.dragging_point] = (x, y)
            self._update_lut()
            self._draw_curve()
            
    def _on_release(self, event):
        self.dragging_point = None
        
    def _on_right_click(self, event):
        idx = self._find_point(event.x, event.y)
        if idx is None:
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="Add Control Point", command=lambda: self._add_point(event.x, event.y))
            menu.tk_popup(event.x_root, event.y_root)
            
    def _add_point(self, cx, cy):
        x, y = self._get_image_coords(cx, cy)
        points = self.control_points[self.current_channel]
        
        insert_idx = 0
        while insert_idx < len(points) and points[insert_idx][0] < x:
            insert_idx += 1
            
        if insert_idx > 0 and insert_idx < len(points):
            points.insert(insert_idx, (x, y))
            self._update_lut()
            self._draw_curve()
            
    def _update_lut(self):
        for channel in ['R', 'G', 'B']:
            points = sorted(self.control_points[channel], key=lambda p: p[0])
            xs = np.array([p[0] for p in points])
            ys = np.array([p[1] for p in points])
            
            cs = CubicSpline(xs, ys, bc_type='natural')
            x_new = np.arange(256)
            y_new = cs(x_new)
            
            y_new = np.clip(y_new, 0, 255)
            self.lut[channel] = y_new.astype(np.uint8)
            
    def _draw_curve(self):
        self.canvas.delete("all")
        
        margin = 20
        plot_size = self.canvas_size - 2 * margin
        
        self.canvas.create_rectangle(margin, margin, margin + plot_size, margin + plot_size, outline='gray')
        
        for i in range(0, 256, 32):
            x1, y1 = self._get_canvas_coords(i, 0)
            x2, y2 = self._get_canvas_coords(i, 255)
            self.canvas.create_line(x1, y1, x2, y2, fill='lightgray', dash=(2, 2))
            
            x1, y1 = self._get_canvas_coords(0, i)
            x2, y2 = self._get_canvas_coords(255, i)
            self.canvas.create_line(x1, y1, x2, y2, fill='lightgray', dash=(2, 2))
            
        x1, y1 = self._get_canvas_coords(0, 0)
        x2, y2 = self._get_canvas_coords(255, 255)
        self.canvas.create_line(x1, y1, x2, y2, fill='gray', dash=(4, 4))
        
        points = self.control_points[self.current_channel]
        channel_colors = {'R': 'red', 'G': 'green', 'B': 'blue'}
        
        xs = np.arange(256)
        ys = self.lut[self.current_channel]
        
        canvas_points = []
        for x, y in zip(xs, ys):
            cx, cy = self._get_canvas_coords(x, y)
            canvas_points.append((cx, cy))
            
        for i in range(len(canvas_points) - 1):
            x1, y1 = canvas_points[i]
            x2, y2 = canvas_points[i + 1]
            self.canvas.create_line(x1, y1, x2, y2, fill=channel_colors[self.current_channel], width=2)
            
        for i, (x, y) in enumerate(points):
            cx, cy = self._get_canvas_coords(x, y)
            color = channel_colors[self.current_channel]
            if i == 0 or i == len(points) - 1:
                self.canvas.create_oval(cx - self.point_radius, cy - self.point_radius,
                                         cx + self.point_radius, cy + self.point_radius,
                                         fill=color, outline='black', width=2)
            else:
                self.canvas.create_oval(cx - self.point_radius, cy - self.point_radius,
                                         cx + self.point_radius, cy + self.point_radius,
                                         fill=color, outline='black')
                
    def _reset_current(self):
        self.control_points[self.current_channel] = [(0, 0), (255, 255)]
        self._update_lut()
        self._draw_curve()
        
    def _reset_all(self):
        for channel in ['R', 'G', 'B']:
            self.control_points[channel] = [(0, 0), (255, 255)]
        self._update_lut()
        self._draw_curve()
        
    def _apply(self):
        self.callback(self.lut)
        self.destroy()


class GaussianParamsDialog(tk.Toplevel):
    def __init__(self, parent, callback, initial_kernel=3, initial_sigma=1.0):
        super().__init__(parent)
        self.title("Gaussian Filter Parameters")
        self.callback = callback
        self.kernel_size = initial_kernel
        self.sigma = initial_sigma
        
        self.transient(parent)
        self.grab_set()
        
        self._create_ui()
        
    def _create_ui(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Kernel Size (odd number):", font=('Arial', 10)).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.kernel_var = tk.IntVar(value=self.kernel_size)
        kernel_spin = ttk.Spinbox(main_frame, from_=3, to=15, textvariable=self.kernel_var, width=10, increment=2)
        kernel_spin.grid(row=0, column=1, sticky=tk.W, pady=5, padx=10)
        
        ttk.Label(main_frame, text="Sigma:", font=('Arial', 10)).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.sigma_var = tk.DoubleVar(value=self.sigma)
        sigma_spin = ttk.Spinbox(main_frame, from_=0.1, to=10.0, textvariable=self.sigma_var, width=10, increment=0.1)
        sigma_spin.grid(row=1, column=1, sticky=tk.W, pady=5, padx=10)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="OK", command=self._apply).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=10)
        
    def _apply(self):
        kernel = self.kernel_var.get()
        if kernel % 2 == 0:
            kernel += 1
        self.callback(kernel, self.sigma_var.get())
        self.destroy()


class EdgeEnhanceParamsDialog(tk.Toplevel):
    def __init__(self, parent, callback, initial_method='sobel'):
        super().__init__(parent)
        self.title("Edge Enhancement Parameters")
        self.callback = callback
        self.method = initial_method
        
        self.transient(parent)
        self.grab_set()
        
        self._create_ui()
        
    def _create_ui(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Select Edge Detection Method:", font=('Arial', 10, 'bold')).pack(pady=10)
        
        self.method_var = tk.StringVar(value=self.method)
        ttk.Radiobutton(main_frame, text="Sobel Edge Detection", variable=self.method_var, value='sobel').pack(anchor=tk.W, padx=20, pady=5)
        ttk.Radiobutton(main_frame, text="Canny Edge Detection", variable=self.method_var, value='canny').pack(anchor=tk.W, padx=20, pady=5)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="OK", command=self._apply).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=10)
        
    def _apply(self):
        self.callback(self.method_var.get())
        self.destroy()


class ISPBlock:
    BLOCK_WIDTH = 120
    BLOCK_HEIGHT = 60
    PORT_RADIUS = 8
    
    def __init__(self, canvas, x, y, block_type, params=None):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.block_type = block_type
        self.params = params if params is not None else self._get_default_params()
        
        self.id = None
        self.text_id = None
        self.input_port_id = None
        self.output_port_id = None
        
        self._draw()
        
    def _get_default_params(self):
        if self.block_type == 'gaussian':
            return {'kernel_size': 3, 'sigma': 1.0}
        elif self.block_type == 'gamma':
            return {'lut': None}
        elif self.block_type == 'edge_enhance':
            return {'method': 'sobel'}
        return {}
        
    def _get_block_color(self):
        colors = {
            'gaussian': '#FFE4B5',
            'gamma': '#98FB98',
            'edge_enhance': '#B0E0E6'
        }
        return colors.get(self.block_type, '#FFFFFF')
        
    def _get_block_name(self):
        names = {
            'gaussian': '高斯滤波',
            'gamma': 'Gamma变换',
            'edge_enhance': '边缘增强'
        }
        return names.get(self.block_type, self.block_type)
        
    def _draw(self):
        if self.id:
            self.canvas.delete(self.id)
        if self.text_id:
            self.canvas.delete(self.text_id)
        if self.input_port_id:
            self.canvas.delete(self.input_port_id)
        if self.output_port_id:
            self.canvas.delete(self.output_port_id)
            
        x1 = self.x - self.BLOCK_WIDTH // 2
        y1 = self.y - self.BLOCK_HEIGHT // 2
        x2 = self.x + self.BLOCK_WIDTH // 2
        y2 = self.y + self.BLOCK_HEIGHT // 2
        
        self.id = self.canvas.create_rectangle(x1, y1, x2, y2, fill=self._get_block_color(), outline='black', width=2)
        
        self.text_id = self.canvas.create_text(self.x, self.y, text=self._get_block_name(), font=('Arial', 10, 'bold'))
        
        self.input_port_id = self.canvas.create_oval(
            x1 - self.PORT_RADIUS, self.y - self.PORT_RADIUS,
            x1 + self.PORT_RADIUS, self.y + self.PORT_RADIUS,
            fill='blue', outline='black'
        )
        
        self.output_port_id = self.canvas.create_oval(
            x2 - self.PORT_RADIUS, self.y - self.PORT_RADIUS,
            x2 + self.PORT_RADIUS, self.y + self.PORT_RADIUS,
            fill='green', outline='black'
        )
        
    def get_input_port_pos(self):
        x1 = self.x - self.BLOCK_WIDTH // 2
        return (x1, self.y)
        
    def get_output_port_pos(self):
        x2 = self.x + self.BLOCK_WIDTH // 2
        return (x2, self.y)
        
    def contains_point(self, x, y):
        x1 = self.x - self.BLOCK_WIDTH // 2
        y1 = self.y - self.BLOCK_HEIGHT // 2
        x2 = self.x + self.BLOCK_WIDTH // 2
        y2 = self.y + self.BLOCK_HEIGHT // 2
        return x1 <= x <= x2 and y1 <= y <= y2
        
    def contains_input_port(self, x, y):
        px, py = self.get_input_port_pos()
        dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
        return dist <= self.PORT_RADIUS + 5
        
    def contains_output_port(self, x, y):
        px, py = self.get_output_port_pos()
        dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
        return dist <= self.PORT_RADIUS + 5


class ISPConnection:
    def __init__(self, canvas, from_block, to_block):
        self.canvas = canvas
        self.from_block = from_block
        self.to_block = to_block
        self.id = None
        
        self._draw()
        
    def _draw(self):
        if self.id:
            self.canvas.delete(self.id)
            
        x1, y1 = self.from_block.get_output_port_pos()
        x2, y2 = self.to_block.get_input_port_pos()
        
        mid_x = (x1 + x2) // 2
        
        self.id = self.canvas.create_line(
            x1, y1, mid_x, y1, mid_x, y2, x2, y2,
            smooth=True, fill='black', width=2
        )
        
    def contains_point(self, x, y, threshold=10):
        x1, y1 = self.from_block.get_output_port_pos()
        x2, y2 = self.to_block.get_input_port_pos()
        
        mid_x = (x1 + x2) // 2
        
        if y1 - threshold <= y <= y1 + threshold and min(x1, mid_x) <= x <= max(x1, mid_x):
            return True
        if mid_x - threshold <= x <= mid_x + threshold and min(y1, y2) <= y <= max(y1, y2):
            return True
        if y2 - threshold <= y <= y2 + threshold and min(mid_x, x2) <= x <= max(mid_x, x2):
            return True
        return False


class ISPEditorWindow(tk.Toplevel):
    def __init__(self, parent, callback, initial_blocks=None, initial_connections=None):
        super().__init__(parent)
        self.title("ISP Adjustment - Block Diagram Editor")
        self.callback = callback
        
        self.blocks = []
        self.connections = []
        
        self.dragging_block = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        
        self.connecting_from = None
        self.temp_line_id = None
        
        self._create_ui()
        
        if initial_blocks:
            self._load_initial_blocks(initial_blocks, initial_connections)
        else:
            self._create_default_blocks()
            
    def _create_ui(self):
        self.geometry("800x500")
        
        toolbar = ttk.Frame(self, padding=5)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(toolbar, text="Add Block:").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(toolbar, text="高斯滤波", command=lambda: self._add_block('gaussian')).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Gamma变换", command=lambda: self._add_block('gamma')).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="边缘增强", command=lambda: self._add_block('edge_enhance')).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Button(toolbar, text="Apply ISP Settings", command=self._apply).pack(side=tk.RIGHT, padx=10)
        
        info_frame = ttk.Frame(self, padding=5)
        info_frame.pack(fill=tk.X)
        ttk.Label(info_frame, text="Instructions: Drag blocks to move | Double-click to edit params | Click output port then input port to connect | Right-click connection to disconnect", foreground='gray').pack()
        
        self.canvas = tk.Canvas(self, bg='white', highlightthickness=1)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        
    def _create_default_blocks(self):
        x_start = 150
        y_mid = 200
        spacing = 180
        
        gaussian = ISPBlock(self.canvas, x_start, y_mid, 'gaussian')
        gamma = ISPBlock(self.canvas, x_start + spacing, y_mid, 'gamma')
        edge = ISPBlock(self.canvas, x_start + 2 * spacing, y_mid, 'edge_enhance')
        
        self.blocks.extend([gaussian, gamma, edge])
        
        self._create_connection(gaussian, gamma)
        self._create_connection(gamma, edge)
        
    def _load_initial_blocks(self, blocks_data, connections_data):
        for block_data in blocks_data:
            block = ISPBlock(self.canvas, block_data['x'], block_data['y'], block_data['type'], block_data['params'])
            self.blocks.append(block)
            
        if connections_data:
            block_map = {i: self.blocks[i] for i in range(len(self.blocks))}
            for from_idx, to_idx in connections_data:
                if from_idx in block_map and to_idx in block_map:
                    self._create_connection(block_map[from_idx], block_map[to_idx])
        
    def _add_block(self, block_type):
        x = self.canvas.winfo_width() // 2
        if x <= 1:
            x = 400
        y = self.canvas.winfo_height() // 2
        if y <= 1:
            y = 250
            
        block = ISPBlock(self.canvas, x, y, block_type)
        self.blocks.append(block)
        
    def _create_connection(self, from_block, to_block):
        if from_block == to_block:
            return
            
        for conn in self.connections:
            if conn.from_block == from_block and conn.to_block == to_block:
                return
                
        connection = ISPConnection(self.canvas, from_block, to_block)
        self.connections.append(connection)
        
    def _on_left_click(self, event):
        x, y = event.x, event.y
        
        if self.connecting_from is not None:
            for block in self.blocks:
                if block.contains_input_port(x, y):
                    if block != self.connecting_from:
                        self._create_connection(self.connecting_from, block)
                    self.connecting_from = None
                    if self.temp_line_id:
                        self.canvas.delete(self.temp_line_id)
                        self.temp_line_id = None
                    return
                    
            self.connecting_from = None
            if self.temp_line_id:
                self.canvas.delete(self.temp_line_id)
                self.temp_line_id = None
            return
            
        for block in self.blocks:
            if block.contains_output_port(x, y):
                self.connecting_from = block
                px, py = block.get_output_port_pos()
                self.temp_line_id = self.canvas.create_line(px, py, x, y, fill='blue', dash=(4, 4))
                return
                
        for block in self.blocks:
            if block.contains_point(x, y):
                self.dragging_block = block
                self.drag_offset_x = x - block.x
                self.drag_offset_y = y - block.y
                return
                
    def _on_drag(self, event):
        x, y = event.x, event.y
        
        if self.dragging_block:
            self.dragging_block.x = x - self.drag_offset_x
            self.dragging_block.y = y - self.drag_offset_y
            self.dragging_block._draw()
            
            for conn in self.connections:
                conn._draw()
            return
            
        if self.connecting_from and self.temp_line_id:
            px, py = self.connecting_from.get_output_port_pos()
            self.canvas.coords(self.temp_line_id, px, py, x, y)
            
    def _on_release(self, event):
        self.dragging_block = None
        
    def _on_double_click(self, event):
        x, y = event.x, event.y
        
        for block in self.blocks:
            if block.contains_point(x, y):
                self._edit_block_params(block)
                return
                
    def _on_right_click(self, event):
        x, y = event.x, event.y
        
        for conn in self.connections:
            if conn.contains_point(x, y):
                menu = tk.Menu(self, tearoff=0)
                menu.add_command(label="Disconnect", command=partial(self._disconnect, conn))
                menu.tk_popup(event.x_root, event.y_root)
                return
                
    def _disconnect(self, connection):
        if connection.id:
            self.canvas.delete(connection.id)
        if connection in self.connections:
            self.connections.remove(connection)
            
    def _edit_block_params(self, block):
        if block.block_type == 'gaussian':
            def on_gaussian_params(kernel_size, sigma):
                block.params['kernel_size'] = kernel_size
                block.params['sigma'] = sigma
                
            dialog = GaussianParamsDialog(self, on_gaussian_params, block.params['kernel_size'], block.params['sigma'])
            self.wait_window(dialog)
            
        elif block.block_type == 'gamma':
            def on_gamma_lut(lut):
                block.params['lut'] = lut
                
            dialog = GammaCurveEditor(self, on_gamma_lut, block.params['lut'])
            self.wait_window(dialog)
            
        elif block.block_type == 'edge_enhance':
            def on_edge_method(method):
                block.params['method'] = method
                
            dialog = EdgeEnhanceParamsDialog(self, on_edge_method, block.params['method'])
            self.wait_window(dialog)
            
    def _get_processing_order(self):
        if not self.blocks:
            return []
            
        block_set = set(self.blocks)
        result = []
        
        in_degree = {block: 0 for block in self.blocks}
        out_edges = {block: [] for block in self.blocks}
        
        for conn in self.connections:
            in_degree[conn.to_block] += 1
            out_edges[conn.from_block].append(conn.to_block)
            
        queue = [block for block in self.blocks if in_degree[block] == 0]
        
        while queue:
            block = queue.pop(0)
            result.append(block)
            
            for neighbor in out_edges[block]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        return result
        
    def _apply(self):
        processing_order = self._get_processing_order()
        blocks_data = []
        connections_data = []
        
        for block in processing_order:
            blocks_data.append({
                'x': block.x,
                'y': block.y,
                'type': block.block_type,
                'params': block.params.copy()
            })
            
        block_indices = {block: i for i, block in enumerate(processing_order)}
        for conn in self.connections:
            if conn.from_block in block_indices and conn.to_block in block_indices:
                connections_data.append((block_indices[conn.from_block], block_indices[conn.to_block]))
                
        self.callback(blocks_data, connections_data)
        self.destroy()


class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor - ISP & Morphology")
        
        self.original_image = None
        self.processed_image = None
        self.gamma_lut = None
        self.threshold_value = 128
        
        self.isp_blocks = None
        self.isp_connections = None
        
        self._orig_tk_image = None
        self._proc_tk_image = None
        
        self._create_ui()
        
    def _create_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        image_frame = ttk.LabelFrame(main_frame, text="Images", padding=5)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        orig_frame = ttk.LabelFrame(image_frame, text="Original", padding=5)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.orig_canvas = tk.Canvas(orig_frame, width=400, height=400, bg='lightgray')
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)
        
        proc_frame = ttk.LabelFrame(image_frame, text="Processed", padding=5)
        proc_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.proc_canvas = tk.Canvas(proc_frame, width=400, height=400, bg='lightgray')
        self.proc_canvas.pack(fill=tk.BOTH, expand=True)
        
        button_frame = ttk.LabelFrame(main_frame, text="Operations", padding=5)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Load Image", command=self._load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="ISP Settings", command=self._open_isp_editor).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="ISP Apply", command=self._apply_isp).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Threshold", command=self._apply_threshold).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Morphology", command=self._open_morphology_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", command=self._reset).pack(side=tk.RIGHT, padx=5)
        
        self.status_var = tk.StringVar(value="Ready. Load an image to begin.")
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, pady=(5, 0))
        
    def _load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                self.original_image = Image.open(file_path).convert('RGB')
                self.processed_image = self.original_image.copy()
                self._display_images()
                self.status_var.set(f"Loaded: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
                
    def _display_images(self):
        if self.original_image:
            self._orig_tk_image = self._display_on_canvas(self.original_image, self.orig_canvas)
            
        if self.processed_image:
            self._proc_tk_image = self._display_on_canvas(self.processed_image, self.proc_canvas)
            
    def _display_on_canvas(self, image, canvas):
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = 400
        if canvas_height <= 1:
            canvas_height = 400
            
        img_width, img_height = image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized)
        
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=tk_image, anchor=tk.CENTER)
        
        return tk_image
        
    def _open_isp_editor(self):
        def on_isp_settings(blocks_data, connections_data):
            self.isp_blocks = blocks_data
            self.isp_connections = connections_data
            self.status_var.set("ISP settings saved. Click 'ISP Apply' to process.")
            
        editor = ISPEditorWindow(self.root, on_isp_settings, self.isp_blocks, self.isp_connections)
        self.root.wait_window(editor)
        
    def _apply_isp(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
            
        if not self.isp_blocks:
            messagebox.showwarning("Warning", "No ISP blocks configured. Please open ISP Settings first.")
            return
            
        result = np.array(self.original_image.copy())
        height, width = result.shape[:2]
        
        for block_data in self.isp_blocks:
            block_type = block_data['type']
            params = block_data['params']
            
            if block_type == 'gaussian':
                if not MORPHOLOGY_CPP_AVAILABLE:
                    messagebox.showerror("Error", "C++ module not available for Gaussian filter.")
                    return
                    
                kernel_size = params['kernel_size']
                sigma = params['sigma']
                
                flat_image = result.flatten()
                filtered = morphology_cpp.gaussian_filter_rgb(flat_image, width, height, kernel_size, sigma)
                result = np.array(filtered).reshape(height, width, 3)
                
            elif block_type == 'gamma':
                lut = params['lut']
                if lut is None:
                    continue
                    
                r = lut['R'][result[:, :, 0]]
                g = lut['G'][result[:, :, 1]]
                b = lut['B'][result[:, :, 2]]
                result = np.stack([r, g, b], axis=2)
                
            elif block_type == 'edge_enhance':
                method = params['method']
                gray = np.dot(result[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                
                if method == 'sobel':
                    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                    
                    from scipy import ndimage
                    grad_x = ndimage.convolve(gray.astype(float), sobel_x)
                    grad_y = ndimage.convolve(gray.astype(float), sobel_y)
                    
                    edge = np.sqrt(grad_x**2 + grad_y**2)
                    edge = np.clip(edge, 0, 255).astype(np.uint8)
                    
                elif method == 'canny':
                    from scipy import ndimage
                    
                    blurred = ndimage.gaussian_filter(gray, sigma=1.0)
                    
                    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                    
                    grad_x = ndimage.convolve(blurred.astype(float), sobel_x)
                    grad_y = ndimage.convolve(blurred.astype(float), sobel_y)
                    
                    magnitude = np.sqrt(grad_x**2 + grad_y**2)
                    direction = np.arctan2(grad_y, grad_x)
                    
                    direction = np.round(direction / (np.pi / 4)) * (np.pi / 4)
                    
                    suppressed = np.zeros_like(magnitude)
                    for i in range(1, magnitude.shape[0] - 1):
                        for j in range(1, magnitude.shape[1] - 1):
                            angle = direction[i, j]
                            if angle == 0 or angle == np.pi or angle == -np.pi:
                                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                            elif angle == np.pi/4 or angle == -3*np.pi/4:
                                neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                            elif angle == np.pi/2 or angle == -np.pi/2:
                                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                            else:
                                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
                                
                            if magnitude[i, j] >= max(neighbors):
                                suppressed[i, j] = magnitude[i, j]
                                
                    high_thresh = 0.2 * suppressed.max()
                    low_thresh = 0.1 * suppressed.max()
                    
                    strong_edges = (suppressed >= high_thresh).astype(np.uint8) * 255
                    weak_edges = ((suppressed >= low_thresh) & (suppressed < high_thresh)).astype(np.uint8) * 255
                    
                    from scipy.ndimage import binary_dilation
                    struct = np.ones((3, 3))
                    strong_dilated = binary_dilation(strong_edges > 0, structure=struct)
                    edge = ((strong_edges > 0) | ((weak_edges > 0) & strong_dilated)).astype(np.uint8) * 255
                    
                result = np.stack([edge, edge, edge], axis=2)
                
        self.processed_image = Image.fromarray(result.astype(np.uint8))
        self._display_images()
        self.status_var.set("ISP processing completed.")
        
    def _apply_threshold(self):
        if self.processed_image is None:
            if self.original_image is None:
                messagebox.showwarning("Warning", "Please load an image first.")
                return
            self.processed_image = self.original_image.copy()
            
        threshold = simpledialog.askinteger("Threshold", "Enter threshold value (0-255):", initialvalue=self.threshold_value, minvalue=0, maxvalue=255)
        
        if threshold is not None:
            self.threshold_value = threshold
            gray = self.processed_image.convert('L')
            img_array = np.array(gray)
            
            binary = np.where(img_array > threshold, 255, 0).astype(np.uint8)
            self.processed_image = Image.fromarray(binary)
            
            self._display_images()
            self.status_var.set(f"Threshold applied (value: {threshold}).")
            
    def _open_morphology_dialog(self):
        if self.processed_image is None:
            messagebox.showwarning("Warning", "Please apply threshold first.")
            return
            
        if not MORPHOLOGY_CPP_AVAILABLE:
            error_msg = "C++ morphology module is not available.\n\n"
            error_msg += "Please compile the C++ module first.\n\n"
            if _morphology_error:
                error_msg += f"Error details: {_morphology_error}"
            messagebox.showerror("Error", error_msg)
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Morphological Operation")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select Operation:", font=('Arial', 10, 'bold')).pack(pady=10)
        
        operation_var = tk.StringVar(value='erode')
        operations = [
            ("Erode", "erode"),
            ("Dilate", "dilate"),
            ("Open (Erode then Dilate)", "open"),
            ("Close (Dilate then Erode)", "close")
        ]
        
        for text, value in operations:
            ttk.Radiobutton(dialog, text=text, variable=operation_var, value=value).pack(anchor=tk.W, padx=20, pady=2)
            
        kernel_frame = ttk.Frame(dialog)
        kernel_frame.pack(pady=10)
        ttk.Label(kernel_frame, text="Kernel Size:").pack(side=tk.LEFT)
        kernel_var = tk.IntVar(value=3)
        kernel_spin = ttk.Spinbox(kernel_frame, from_=3, to=15, textvariable=kernel_var, width=5, increment=2)
        kernel_spin.pack(side=tk.LEFT, padx=5)
        
        def apply_operation():
            op = operation_var.get()
            kernel_size = kernel_var.get()
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            self._apply_morphology(op, kernel_size)
            dialog.destroy()
            
        ttk.Button(dialog, text="Apply", command=apply_operation).pack(pady=10)
        
        dialog.update_idletasks()
        dialog.minsize(dialog.winfo_width(), dialog.winfo_height())
        
        self.root.wait_window(dialog)
        
    def _apply_morphology(self, operation, kernel_size):
        if self.processed_image is None:
            return
            
        gray = self.processed_image.convert('L')
        img_array = np.array(gray)
        height, width = img_array.shape
        
        flat_image = img_array.flatten()
        
        op_map = {
            'erode': morphology_cpp.MorphologyOp.ERODE,
            'dilate': morphology_cpp.MorphologyOp.DILATE,
            'open': morphology_cpp.MorphologyOp.OPEN,
            'close': morphology_cpp.MorphologyOp.CLOSE
        }
        
        result = morphology_cpp.morphology_operation(flat_image, width, height, op_map[operation], kernel_size)
        result_array = np.array(result).reshape(height, width)
            
        self.processed_image = Image.fromarray(result_array.astype(np.uint8))
        
        self._display_images()
        op_names = {'erode': 'Erosion', 'dilate': 'Dilation', 'open': 'Opening', 'close': 'Closing'}
        self.status_var.set(f"{op_names[operation]} applied (kernel: {kernel_size}x{kernel_size}, backend: C++).")
        
    def _reset(self):
        if self.original_image:
            self.processed_image = self.original_image.copy()
            self.gamma_lut = None
            self._display_images()
            self.status_var.set("Reset to original image.")


def main():
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
