class AlgorithmWrapper():

    def embed(self, image, watermark_data):
        raise NotImplementedError
    
    def extract(self, image, watermark_data):
        raise NotImplementedError
