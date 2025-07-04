from typing import (
    Dict,
    Any
)


def planarize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    may_be_dict_inside = True
    while may_be_dict_inside:
        may_be_dict_inside = False
        for k in list(d.keys()):
            v = d[k]
            if isinstance(v, dict):
                may_be_dict_inside = True
                for k_ in list(v.keys()):
                    v_ = v[k_]
                    plain_key = "_".join([k, k_])
                    d[plain_key] = v_
                del d[k]
    return d