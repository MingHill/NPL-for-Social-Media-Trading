# llm.py
class BaseCodeSim:
    """
    A base interface for Large Language Models.
    Subclasses should at least implement:
      - chat(prompt) -> str
      - chat_stream(prompt) -> Generator[str, None, None]
    """
    def __init__(self, code):
        self.code = code
        self.target_code = None
                
    def code_similarity(self, code1, code2, chunk_size=30, window_size=4):
        """
        Returns a percentage indicating how similar the two code strings are,
        using CopyDetect under the hood.
        
        Requirements:
            pip install copydetect
        
        Parameters:
            code1 (str): First code snippet as a string.
            code2 (str): Second code snippet as a string.
            chunk_size (int): Number of tokens in each chunk for fingerprinting.
            window_size (int): Sliding window size for building winnowed fingerprints.

        Returns:
            float: Highest similarity percentage between the two code snippets.
        """
        import tempfile
        import copydetect

        # If either snippet is empty, similarity is 0
        if not code1 or not code2:
            return 0.0

        # Write each snippet to its own temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp1:
            tmp1.write(code1.encode('utf-8'))
            file1 = tmp1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp2:
            tmp2.write(code2.encode('utf-8'))
            file2 = tmp2.name

        # Build the fingerprints
        fp1 = copydetect.CodeFingerprint(file1, chunk_size, window_size)
        fp2 = copydetect.CodeFingerprint(file2, chunk_size, window_size)

        # Compare them
        token_overlap, similarities, slices = copydetect.compare_files(fp1, fp2)
        
        # Return the highest similarity as a percentage
        if similarities:
            return round(max(similarities) * 100, 2)
        return 0.0

    def is_similar(self) -> bool:
        """
        Returns True if the response is similar to the prompt.
        """
        similarity = self.code_similarity(self.target_code, self.code)
        return similarity == 100.0
    
class SafeCodeSim(BaseCodeSim):
    """
    A safe interface for Large Language Models.
    Subclasses should at least implement:
      - chat(prompt) -> str
    """
    def __init__(self, code):
        super().__init__(code)
        self.target_address = "0xD015aC4B47Fe34D61F732FA4B493fDB09e7C1471"
        # load target code from data/code/address
        self.target_code = None
        with open(f"data/code/{self.target_address}.txt", "r") as f:
            self.target_code = f.read()


