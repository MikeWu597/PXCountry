import cv2
import os


def select_roi(img, window_name):
    roi = [None, None, None, None]  # x, y, w, h
    drawing = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, roi
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            roi[0], roi[1] = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                roi[2] = x - roi[0]
                roi[3] = y - roi[1]
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi[2] = x - roi[0]
            roi[3] = y - roi[1]

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        img_copy = img.copy()
        if roi[0] is not None and roi[2] is not None:
            cv2.rectangle(img_copy, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
        cv2.imshow(window_name, img_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter键确认
            break
        elif key == 27:  # Esc键取消
            roi = [None] * 4
            break

    cv2.destroyWindow(window_name)
    if any(v is None or v <= 0 for v in roi[2:]):
        return None
    return tuple(roi)


def apply_mosaic(img, roi):
    x, y, w, h = roi
    x, y = max(0, x), max(0, y)
    w = min(img.shape[1] - x, w)
    h = min(img.shape[0] - y, h)
    if w <= 0 or h <= 0:
        return img
    region = img[y:y + h, x:x + w]
    region = cv2.resize(cv2.resize(region, (10, 10)), (w, h), interpolation=cv2.INTER_NEAREST)
    img[y:y + h, x:x + w] = region
    return img


def process_images(folder_path):
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
              if os.path.splitext(f)[1].lower() in image_exts]

    for image_path in images:
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取: {image_path}")
            continue

        # 裁剪阶段
        print(f"处理: {image_path}")
        print("选择裁剪区域后按Enter...")
        crop_roi = select_roi(img, "裁剪选择")
        if not crop_roi:
            print("跳过裁剪")
            continue
        x, y, w, h = crop_roi
        cropped_img = img[y:y + h, x:x + w]

        # 马赛克阶段
        print("选择马赛克区域后按Enter...")
        mosaic_roi = select_roi(cropped_img, "马赛克选择")
        if not mosaic_roi:
            print("跳过马赛克")
            continue

        # 应用马赛克并保存
        final_img = apply_mosaic(cropped_img, mosaic_roi)
        cv2.imwrite(image_path, final_img)
        print(f"已保存: {image_path}")


if __name__ == "__main__":
    folder = input("请输入图片文件夹路径: ").strip()
    process_images(folder)