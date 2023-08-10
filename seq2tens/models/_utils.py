def _check_for_hparam_in_dict(hp_dict, hp_name, hp_default=None):
    if hp_dict is not None and hp_name in hp_dict:
        if hp_dict[hp_name] is not None:
            hp_value = hp_dict[hp_name]
            del hp_dict[hp_name]
            return hp_value
        del hp_dict[hp_name]
    return hp_default 
    
def _normalize_list(_maybe_a_list):
    if _maybe_a_list is not None:
        if isinstance(_maybe_a_list, (list, tuple)):
            return list(_maybe_a_list)
        else:
            return [_maybe_a_list] # wasn't