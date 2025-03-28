from utils.common.dict_mappings import CONVERT_PLOTNAME_DICT


def convert_paramname_for_plot(paramname_list):
    convert_paramname_dict = CONVERT_PLOTNAME_DICT
    
    if paramname_list:
        return [ convert_paramname_dict[p_name] if p_name in convert_paramname_dict else None for p_name in paramname_list]
    else:
        return None
