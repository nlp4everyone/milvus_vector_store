# Typing
from typing import Union, List, Tuple
# Coponents
import pathlib, mimetypes, base64
from io import BytesIO

# PIL
from PIL import Image

class ImageUtils():
    @staticmethod
    def get_image_path(file_paths: List[str]) -> Tuple[List[str], List[str]]:
        # Image extension
        img_extensions = [".png", ".jpeg", ".jpg"]

        # Check url, if isnt a file, meaning is url
        contain_files = [pathlib.Path(path).is_file() for path in file_paths]

        if contain_files.count(True) == 0:
            # When all string is url link, return both file path as variables
            return file_paths, file_paths

        # Filter only image url (Full path)
        images_url = [path for path in file_paths if pathlib.Path(path).suffix in img_extensions]
        if len(images_url) == 0:
            raise FileNotFoundError("No images from input!")

        # Filter only image path (Only name)
        images_name = [pathlib.Path(url).name for url in images_url]
        # Return
        return images_url, images_name

    @staticmethod
    def get_image_mimetype(images: List[Union[str,Image.Image]]) -> List[Union[str, None]]:
        # Image path case
        if isinstance(images[0],str):
            return [mimetypes.guess_type(pathlib.Path(image_path))[0] for image_path in images]
        # PIL Image case
        return [Image.MIME.get(image.format) for image in images]

    @staticmethod
    def image_to_base64(image :Image.Image):
        # Create a BytesIO object to hold the image data
        buffered = BytesIO()
        # Save the image to the BytesIO object in PNG format (you can change this)
        image.save(buffered, format="PNG")
        # Get the byte data and encode it to Base64
        return base64.b64encode(buffered.getvalue()).decode("utf-8")