import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from scipy.interpolate import CubicSpline
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'morphology'))

try:
    import morphology_cpp
    HAS_CPP_MODULE = True
except (ImportError, ValueError, RuntimeError) as e:
    HAS_CPP_MODULE = False
    print(f"Warning: C++ morphology module not available ({type(e).__name__}: {e}). Using fallback Python implementation.")


class GammaCurveEditor(tk.Toplevel):
    def __init__(self, parent, callback):
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


class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor - Gamma & Morphology")
        
        self.original_image = None
        self.processed_image = None
        self.gamma_lut = None
        self.threshold_value = 128
        
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
        ttk.Button(button_frame, text="Gamma Adjust", command=self._open_gamma_editor).pack(side=tk.LEFT, padx=5)
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
            self._display_on_canvas(self.original_image, self.orig_canvas)
            
        if self.processed_image:
            self._display_on_canvas(self.processed_image, self.proc_canvas)
            
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
        self._tk_image = ImageTk.PhotoImage(resized)
        
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=self._tk_image, anchor=tk.CENTER)
        
    def _open_gamma_editor(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
            
        editor = GammaCurveEditor(self.root, self._apply_gamma)
        self.root.wait_window(editor)
        
    def _apply_gamma(self, lut):
        if self.original_image is None:
            return
            
        self.gamma_lut = lut
        img_array = np.array(self.original_image)
        
        r = lut['R'][img_array[:, :, 0]]
        g = lut['G'][img_array[:, :, 1]]
        b = lut['B'][img_array[:, :, 2]]
        
        result = np.stack([r, g, b], axis=2)
        self.processed_image = Image.fromarray(result.astype(np.uint8))
        
        self._display_images()
        self.status_var.set("Gamma correction applied.")
        
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
            
        dialog = tk.Toplevel(self.root)
        dialog.title("Morphological Operation")
        dialog.geometry("300x200")
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
            ttk.Radiobutton(dialog, text=text, variable=operation_var, value=value).pack(anchor=tk.W, padx=20)
            
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
        
        self.root.wait_window(dialog)
        
    def _apply_morphology(self, operation, kernel_size):
        if self.processed_image is None:
            return
            
        gray = self.processed_image.convert('L')
        img_array = np.array(gray)
        height, width = img_array.shape
        
        flat_image = img_array.flatten()
        
        op_map = {
            'erode': morphology_cpp.MorphologyOp.ERODE if HAS_CPP_MODULE else None,
            'dilate': morphology_cpp.MorphologyOp.DILATE if HAS_CPP_MODULE else None,
            'open': morphology_cpp.MorphologyOp.OPEN if HAS_CPP_MODULE else None,
            'close': morphology_cpp.MorphologyOp.CLOSE if HAS_CPP_MODULE else None
        }
        
        if HAS_CPP_MODULE:
            result = morphology_cpp.morphology_operation(flat_image, width, height, op_map[operation], kernel_size)
            result_array = np.array(result).reshape(height, width)
        else:
            result_array = self._python_morphology(img_array, operation, kernel_size)
            
        self.processed_image = Image.fromarray(result_array.astype(np.uint8))
        
        self._display_images()
        op_names = {'erode': 'Erosion', 'dilate': 'Dilation', 'open': 'Opening', 'close': 'Closing'}
        backend = "C++" if HAS_CPP_MODULE else "Python"
        self.status_var.set(f"{op_names[operation]} applied (kernel: {kernel_size}x{kernel_size}, backend: {backend}).")
        
    def _python_morphology(self, img_array, operation, kernel_size):
        half = kernel_size // 2
        height, width = img_array.shape
        result = np.zeros_like(img_array)
        
        if operation == 'erode':
            for y in range(height):
                for x in range(width):
                    min_val = 255
                    for ky in range(-half, half + 1):
                        for kx in range(-half, half + 1):
                            py = y + ky
                            px = x + kx
                            if 0 <= py < height and 0 <= px < width:
                                min_val = min(min_val, img_array[py, px])
                    result[y, x] = min_val
                    
        elif operation == 'dilate':
            for y in range(height):
                for x in range(width):
                    max_val = 0
                    for ky in range(-half, half + 1):
                        for kx in range(-half, half + 1):
                            py = y + ky
                            px = x + kx
                            if 0 <= py < height and 0 <= px < width:
                                max_val = max(max_val, img_array[py, px])
                    result[y, x] = max_val
                    
        elif operation == 'open':
            eroded = self._python_morphology(img_array, 'erode', kernel_size)
            result = self._python_morphology(eroded, 'dilate', kernel_size)
            
        elif operation == 'close':
            dilated = self._python_morphology(img_array, 'dilate', kernel_size)
            result = self._python_morphology(dilated, 'erode', kernel_size)
            
        return result
        
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
