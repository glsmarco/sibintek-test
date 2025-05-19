from environment import Environment
from PIL import Image


def init_image_preprocess() -> None:
    environment = Environment()
    img_save_counter = 0
    for img_path in environment.raw_images_dir.iterdir():
        with Image.open(img_path) as img:
            w, h = img.size 
            target_size = environment.config.target_size
            if w < target_size: # image is too small to read the text
                continue
            if h > w: # some strange vehicle passport format
                continue
            img: Image.Image = resize_and_center_on_black(img, target_width=target_size, canvas_size=target_size) # width -> target_size, height is padded if needed
            dst_path = environment.standartized_images_dir / f"sample_{img_save_counter}.jpg"
            dst_path.parent.mkdir(exist_ok=True, parents=True)
            img.save(
                dst_path
            )
            img_save_counter += 1 # we use separate counter as some init samples are skipped due to low quality


def resize_and_center_on_black(image: Image.Image, target_width: int, canvas_size: int) -> Image.Image:
    """
    Resize an image to a target width while preserving aspect ratio,
    and paste it centered on a black square canvas of size canvas_size × canvas_size.

    Args:
        image (Image.Image): Input PIL image.
        target_width (int): Width to resize image to.
        canvas_size (int): Size of the square canvas.

    Returns:
        Image.Image: Final image centered on a black square canvas.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL.Image.Image instance.")
    
    if target_width > canvas_size:
        raise ValueError("Target width cannot be larger than canvas size.")
    
    orig_w, orig_h = image.size
    aspect_ratio = orig_h / orig_w
    new_height = round(target_width * aspect_ratio)
    
    resized = image.resize((target_width, new_height), resample=Image.Resampling.LANCZOS)
    canvas = Image.new(mode=resized.mode, size=(canvas_size, canvas_size), color=0)

    # Compute top-left paste position
    offset_x = (canvas_size - target_width) // 2
    offset_y = (canvas_size - new_height) // 2

    canvas.paste(resized, (offset_x, offset_y))
    return canvas