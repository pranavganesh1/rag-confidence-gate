class Chunker:
    def __init__(self, chunk_size=512, overlap=0.2):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be between 0 and 1")

        self.chunk_size = chunk_size
        self.step = max(1, int(chunk_size * (1 - overlap)))

    def chunk(self, text):
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.step):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk:
                chunks.append(chunk)

            if i + self.chunk_size >= len(words):
                break

        return chunks
