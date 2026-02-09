def h5_tree(filename, *, show_attrs=False, max_depth=None, file_mode="r") -> str:
    """
    Return a plain-text tree of an HDF5 file and optionally save to .txt.
    - show_attrs: include a compact summary of attributes at each node
    - max_depth:  limit recursion depth (None = full)
    """
    import h5py, textwrap, os

    def _attr_summary(obj):
        if not show_attrs or not hasattr(obj, "attrs") or len(obj.attrs) == 0:
            return ""
        keys = list(obj.attrs.keys())
        return f"  (attrs: {len(keys)}: {', '.join(map(str, keys[:10]))}{'...' if len(keys) > 10 else ''})"

    def _node_line(name, obj, is_group):
        if is_group:
            return f"{name}/  [group]{_attr_summary(obj)}"
        shape = getattr(obj, "shape", "?")
        dtype = getattr(obj, "dtype", "?")
        chunks = getattr(obj, "chunks", None)
        comp = None
        try:
            f = obj.compression
            comp = f"{f}" if f else None
        except Exception:
            pass
        extras = []
        if chunks: extras.append(f"chunks={chunks}")
        if comp:   extras.append(f"comp={comp}")
        extra = (", " + ", ".join(extras)) if extras else ""
        return f"{name}  [dset] shape={shape} dtype={dtype}{extra}{_attr_summary(obj)}"

    def _walk(g, prefix="", depth=0):
        lines = []
        if max_depth is not None and depth > max_depth:
            return lines
        items = list(g.items())
        items.sort(key=lambda kv: (0 if isinstance(kv[1], h5py.Group) else 1, kv[0]))
        n = len(items)
        for i, (k, v) in enumerate(items):
            last = (i == n - 1)
            branch = "`-- " if last else "|-- "
            cont   = "    " if last else "|   "
            if isinstance(v, h5py.Group):
                lines.append(prefix + branch + _node_line(k, v, True))
                if max_depth is None or depth < max_depth:
                    lines.extend(_walk(v, prefix + cont, depth + 1))
            elif isinstance(v, h5py.Dataset):
                lines.append(prefix + branch + _node_line(k, v, False))
            else:
                lines.append(prefix + branch + f"{k}  [link/other]{_attr_summary(v)}")
        return lines

    with h5py.File(filename, file_mode) as f:
        header = f"/  [file]{_attr_summary(f)}"
        body = "\n".join(_walk(f, "", 1))
    return header + ("\n" + body if body else "")


if __name__ == "__main__":
    import os
    from src.config_params import Config

    cfg = Config()
    fps = [
        cfg.NKD_RAW_FILE,
        cfg.NKD_PROCESSED_FILE,
        cfg.PH_RAW_FILE,
        cfg.PH_PROCESSED_FILE,
        # "/home/masseyj/Downloads/D_shear_SU_production_test.hdf5"
        
    ]
    for fp in fps:
        tree_str = h5_tree(fp, show_attrs=True, max_depth=80)
        out_path = os.path.splitext(fp)[0] + "_tree.txt"
        with open(out_path, "w", encoding="utf-8") as f_out:
            f_out.write(tree_str)
        print(f"Saved: {out_path}")
