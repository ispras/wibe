class RegistryMeta(type):
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
            if "report_name" not in cls.__dict__:
                setattr(cls, "report_name", plugin_name)
            for base in bases:
                if hasattr(base, "_registry"):
                    if plugin_name in base._registry:
                        raise ValueError(f"{plugin_name} already registered")
                    base._registry[plugin_name] = cls
                    print(
                        f"Registered {base.type}: {cls.__name__} as {plugin_name}"
                    )
                    break
