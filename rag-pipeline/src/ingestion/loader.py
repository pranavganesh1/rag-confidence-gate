from pathlib import Path

import fitz


class DocumentLoader:
    def load_pdf(self, path):
        file_path = Path(path)
        with fitz.open(file_path) as doc:
            return " ".join(page.get_text() for page in doc)

    def load_text(self, path):
        file_path = Path(path)
        return file_path.read_text(encoding="utf-8")
