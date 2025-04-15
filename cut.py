import os
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class ImageCropper:
    def __init__(self, root):
        self.root = root
        self.root.title("图片批量裁剪工具")

        # 初始化参数
        self.folder_path = filedialog.askdirectory()
        if not self.folder_path:
            self.root.destroy()
            return

        self.extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        self.image_files = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path)
                            if os.path.splitext(f)[1].lower() in self.extensions]

        if not self.image_files:
            messagebox.showinfo("提示", "文件夹中没有图片文件")
            self.root.destroy()
            return

        self.current_index = 0
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.tk_image = None
        self.original_image = None
        self.scale_x = 1.0
        self.scale_y = 1.0

        # 创建GUI组件
        self.canvas = Canvas(root, bg='gray', cursor='cross')
        self.canvas.pack(fill=BOTH, expand=True)

        self.status_bar = Label(root, text="按Enter确认裁剪，Esc取消选择", bd=1, relief=SUNKEN, anchor=W)
        self.status_bar.pack(side=BOTTOM, fill=X)

        # 绑定事件
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Return>", self.confirm_crop)
        self.root.bind("<Escape>", self.cancel_crop)

        # 加载第一张图片
        self.load_image()

    def load_image(self):
        if self.current_index >= len(self.image_files):
            messagebox.showinfo("完成", "所有图片处理完成！")
            self.root.destroy()
            return

        # 清理画布
        self.canvas.delete("all")
        self.status_bar.config(
            text=f"处理 {self.current_index + 1}/{len(self.image_files)}，按Enter确认裁剪，Esc取消选择")

        # 加载图片并适配窗口
        image_path = self.image_files[self.current_index]
        self.original_image = Image.open(image_path)

        # 计算缩放比例
        img_width, img_height = self.original_image.size
        max_size = 800
        if max(img_width, img_height) > max_size:
            ratio = max_size / max(img_width, img_height)
            new_size = (int(img_width * ratio), int(img_height * ratio))
            displayed_image = self.original_image.resize(new_size, Image.LANCZOS)
            self.scale_x = img_width / new_size[0]
            self.scale_y = img_height / new_size[1]
        else:
            displayed_image = self.original_image.copy()
            self.scale_x = self.scale_y = 1.0

        # 显示图片
        self.tk_image = ImageTk.PhotoImage(displayed_image)
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.create_image(0, 0, anchor=NW, image=self.tk_image)

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)

    def on_drag(self, event):
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline='red', width=2
        )

    def on_release(self, event):
        self.end_x = event.x
        self.end_y = event.y

    def confirm_crop(self, event=None):
        if not all([self.start_x, self.start_y, self.end_x, self.end_y]):
            return

        # 转换坐标到原始尺寸
        x0 = int(min(self.start_x, self.end_x) * self.scale_x)
        y0 = int(min(self.start_y, self.end_y) * self.scale_y)
        x1 = int(max(self.start_x, self.end_x) * self.scale_x)
        y1 = int(max(self.start_y, self.end_y) * self.scale_y)

        try:
            # 裁剪并覆盖保存
            cropped = self.original_image.crop((x0, y0, x1, y1))
            cropped.save(self.image_files[self.current_index])
            self.current_index += 1
            self.load_image()
        except Exception as e:
            messagebox.showerror("错误", f"保存失败：{str(e)}")

    def cancel_crop(self, event=None):
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None


if __name__ == "__main__":
    root = Tk()
    app = ImageCropper(root)
    root.mainloop()