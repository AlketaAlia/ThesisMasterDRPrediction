"""File upload handling: individual images and ZIP archives.

ZIP handling enforces two limits to prevent zip-bomb attacks: the entry count
and the total uncompressed size.
"""
import io
import os
import zipfile

from PIL import Image

from lib.config import MAX_ZIP_FILES, MAX_ZIP_UNCOMPRESSED_BYTES


def load_uploaded_images(uploaded_files, on_warning=None, on_error=None):
    """Load images from individual uploads or ZIP archives.

    Args:
        uploaded_files: iterable of Streamlit UploadedFile objects.
        on_warning, on_error: optional callbacks (e.g. `st.warning` / `st.error`)
            invoked when a ZIP exceeds limits. Decoupling this from Streamlit
            keeps the module unit-testable.

    Streamlit's UploadedFile is not reliably hashable, so this is no longer
    cached — caching was likely a no-op anyway.
    """
    images = []
    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        suffix = os.path.splitext(name)[1].lower()
        file_bytes = uploaded_file.getvalue()
        if suffix == ".zip":
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zip_ref:
                infos = zip_ref.infolist()
                if len(infos) > MAX_ZIP_FILES:
                    if on_warning:
                        on_warning(
                            f"ZIP contains {len(infos)} entries; only the first "
                            f"{MAX_ZIP_FILES} will be processed."
                        )
                    infos = infos[:MAX_ZIP_FILES]
                total_uncompressed = sum(info.file_size for info in infos)
                if total_uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
                    if on_error:
                        on_error(
                            f"ZIP uncompressed size ({total_uncompressed / 1e6:.0f} MB) "
                            f"exceeds the {MAX_ZIP_UNCOMPRESSED_BYTES / 1e6:.0f} MB limit."
                        )
                    continue
                for info in infos:
                    inner_name = info.filename
                    if inner_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        with zip_ref.open(info) as inner_file:
                            image_bytes = inner_file.read()
                            images.append({
                                "name": inner_name,
                                "image": Image.open(io.BytesIO(image_bytes)).convert("RGB"),
                            })
        elif suffix in {".png", ".jpg", ".jpeg"}:
            images.append({
                "name": name,
                "image": Image.open(io.BytesIO(file_bytes)).convert("RGB"),
            })
    return images
