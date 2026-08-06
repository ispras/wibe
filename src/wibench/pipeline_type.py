from enum import Flag, auto


class PipelineType(Flag):
    IMAGE = auto()
    PROMPT = auto()
    ALL = IMAGE | PROMPT
    
    @classmethod
    def single_types(cls):
        return [
            cls.IMAGE,
            cls.PROMPT,
        ]
