from loguru import logger
from wibench.pipeline_type import PipelineType


class RegistryMeta(type):
    """Metaclass for implementing automatic plugin registration systems.
    
    This metaclass automatically registers all concrete classes that inherit from
    a base class using this metaclass. The registration system allows for
    dynamic discovery and instantiation of plugin implementations.
    """
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)

        if not hasattr(cls, "_registry"):
            cls._registry = {}
            return

        if cls.__dict__.get("abstract", False):
            return

        if not cls.__name__.startswith("Base"):
            if "name" in cls.__dict__:
                plugin_name = cls.name
            else:
                plugin_name = cls.__name__
            plugin_name = plugin_name.lower()
            setattr(cls, "report_name", plugin_name)
            if "pipeline_type" not in cls.__dict__:
                setattr(cls, "pipeline_type", PipelineType.ALL)
            for base in bases:
                if hasattr(base, "_registry"):
                    if plugin_name in base._registry:
                        raise ValueError(f"{plugin_name} already registered")
                    base._registry[plugin_name] = cls
                    logger.info(
                        f"Registered {base.type}: {cls.__name__} as {plugin_name}"
                    )
                    break
