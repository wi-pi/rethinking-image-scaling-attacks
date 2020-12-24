def set_ccs_font(fontsize=None):
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = "Linux Libertine"
    mpl.rcParams['font.family'] = "sans-serif"
    if fontsize:
        mpl.rcParams['font.size'] = fontsize
