def set_ccs_font(fontsize=None):
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = "Linux Libertine"
    mpl.rcParams['font.family'] = "sans-serif"
    if fontsize:
        mpl.rcParams['font.size'] = fontsize


def get_id_list_by_ratio(id_list, ratio):
    if ratio == 2:
        return id_list[::15]
    if ratio == 3:
        return id_list[::2] + id_list[::5]
    if ratio == 4:
        return id_list
    if ratio == 5:
        return id_list
    raise NotImplementedError
