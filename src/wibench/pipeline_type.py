from enum import Flag, auto


class PipelineType(Flag):
    IMAGE = auto()
    PROMPT = auto()
    AUDIO = auto()
    ALL_IMAGE = IMAGE | PROMPT
    ALL = IMAGE | PROMPT | AUDIO
    
    @classmethod
    def single_types(cls):
        return [
            cls.IMAGE,
            cls.PROMPT,
            cls.AUDIO,
        ]
