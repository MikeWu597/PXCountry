import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class MosaicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片马赛克工具")

        # 图片列表和当前索引
        self.image_list = []
        self.current_index = 0

        # 当前图片和显示对象
        self.current_image = None
        self.photo = None

        # 画布和滚动条
        self.canvas = tk.Canvas(root)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 添加滚动条
        self.scrollbar_v = tk.Scrollbar(root, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
        self.scrollbar_h = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.configure(yscrollcommand=self.scrollbar_v.set, xscrollcommand=self.scrollbar_h.set)

        # 绑定鼠标事件
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # 矩形区域列表和当前绘制的矩形
        self.rectangles = []
        self.current_rect = None

        # 控制按钮
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.TOP, pady=10)

        self.load_button = tk.Button(self.control_frame, text="选择文件夹", command=self.load_folder)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.apply_button = tk.Button(self.control_frame, text="应用马赛克并下一张", command=self.apply_and_next)
        self.apply_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(self.control_frame, text="清除所有区域", command=self.clear_rectangles)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # 状态栏
        self.status_label = tk.Label(root, text="未选择文件夹", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def load_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return

        # 获取所有图片文件
        self.image_list = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

        if not self.image_list:
            messagebox.showerror("错误", "文件夹中没有图片文件")
            return

        self.current_index = 0
        self.load_image()

    def load_image(self):
        if self.current_index >= len(self.image_list):
            messagebox.showinfo("完成", "所有图片处理完成")
            self.root.quit()
            return

        image_path = self.image_list[self.current_index]
        self.current_image = Image.open(image_path)

        # 调整画布滚动区域
        self.canvas.config(scrollregion=(0, 0, self.current_image.width, self.current_image.height))

        # 显示图片
        self.photo = ImageTk.PhotoImage(self.current_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # 重置矩形列表
        self.rectangles = []
        self.status_label.config(text=f"处理中: {self.current_index + 1}/{len(self.image_list)}")

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        if self.current_rect:
            self.canvas.delete(self.current_rect)

        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='red', width=2
        )

    def on_mouse_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.current_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)

        # 调整坐标顺序
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)

        # 限制在图片范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.current_image.width, x2)
        y2 = min(self.current_image.height, y2)

        self.rectangles.append((x1, y1, x2, y2))
        self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)
        self.current_rect = None

    def apply_and_next(self):
        if not self.rectangles:
            messagebox.showwarning("警告", "请至少选择一个区域")
            return

        # 应用马赛克
        image = self.current_image.copy()
        block_size = 10  # 马赛克块大小

        for rect in self.rectangles:
            x1, y1, x2, y2 = map(int, rect)
            region = image.crop((x1, y1, x2, y2))

            # 计算缩放比例
            w, h = region.size
            if w == 0 or h == 0:
                continue
            small = region.resize((block_size, block_size), Image.NEAREST)
            mosaic = small.resize((w, h), Image.NEAREST)

            image.paste(mosaic, (x1, y1, x2, y2))

        # 保存图片（覆盖原文件）
        try:
            image.save(self.image_list[self.current_index], quality=90)
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
            return

        # 加载下一张图片
        self.current_index += 1
        self.load_image()

    def clear_rectangles(self):
        self.rectangles = []
        self.canvas.delete("all")
        self.photo = ImageTk.PhotoImage(self.current_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)


if __name__ == "__main__":
    root = tk.Tk()
    app = MosaicApp(root)
    root.mainloop()